"""Merge Roboflow YOLO exports into one clean, leak-free detection dataset.

Steps:
  1. Remap each source's class names to canonical classes (unmapped boxes dropped).
  2. Group Roboflow augmentation copies by original filename stem.
  3. Merge groups whose images are near-duplicates (dHash), across sources too.
  4. Split by group (no group spans two splits). Val/test keep one image per group.

Usage:
  python tools/build_dataset.py --config tools/configs/idr_notes.yaml
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict

import numpy as np
import yaml
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}$")


def dhash(path, size=16):
    """Difference hash as packed bytes (size*size bits)."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def read_labels(path, remap):
    """Return list of (class_name, cx, cy, w, h). Polygons become their bbox."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            cls = remap.get(int(p[0]))
            if cls is None:
                continue
            v = list(map(float, p[1:]))
            if len(v) == 4:
                cx, cy, w, h = v
            else:
                xs, ys = v[0::2], v[1::2]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            if w > 0 and h > 0:
                out.append((cls, cx, cy, w, h))
    return out


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def collect(cfg):
    items = []
    for src in cfg["sources"]:
        root = os.path.join(cfg["raw_dir"], src["dir"])
        if not os.path.isdir(root):
            print(f"WARNING: missing source {src['dir']}, skipped")
            continue
        if os.path.exists(os.path.join(root, "data.yaml")):
            with open(os.path.join(root, "data.yaml")) as f:
                names = yaml.safe_load(f)["names"]
        else:  # Label Studio style export: classes.txt next to images/ and labels/
            with open(os.path.join(root, "classes.txt")) as f:
                names = [ln.strip() for ln in f if ln.strip()]
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        remap = {i: src["classes"].get(str(n)) for i, n in enumerate(names)}
        # Optional regex removed from the stem so e.g. frames of one video share a group.
        strip = re.compile(src["group_strip"]) if src.get("group_strip") else None
        for dirpath, _, files in os.walk(root):
            if os.path.basename(dirpath) != "images":
                continue
            lbl_dir = os.path.join(os.path.dirname(dirpath), "labels")
            for fn in files:
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in IMG_EXT:
                    continue
                boxes = read_labels(os.path.join(lbl_dir, stem + ".txt"), remap)
                if not boxes:
                    continue
                key = RF_SUFFIX.sub("", stem)
                if strip:
                    key = strip.sub("", key)
                items.append({
                    "img": os.path.join(dirpath, fn),
                    "source": src["dir"],
                    "group_key": src["dir"] + "/" + key,
                    "boxes": boxes,
                })
    return items


def load_hashes(items, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    hashes = []
    for it in items:
        k = os.path.relpath(it["img"])
        if k not in cache:
            cache[k] = dhash(it["img"]).tobytes().hex()
        hashes.append(np.frombuffer(bytes.fromhex(cache[k]), dtype=np.uint8))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return np.stack(hashes)


def build_groups(items, max_dist, cache_path):
    keys = sorted({it["group_key"] for it in items})
    kid = {k: i for i, k in enumerate(keys)}
    uf = UnionFind(len(keys))
    hb = load_hashes(items, cache_path)
    gids = np.array([kid[it["group_key"]] for it in items])
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    for i in range(len(items)):
        d = lut[np.bitwise_xor(hb[i + 1:], hb[i])].sum(axis=1)
        for j in np.nonzero(d <= max_dist)[0] + i + 1:
            uf.union(gids[i], gids[j])
    for it in items:
        it["group"] = uf.find(kid[it["group_key"]])
    return items


def cap_per_key(its, n):
    """Keep at most n evenly spaced images per original/video key."""
    if not n:
        return its
    by_key = defaultdict(list)
    for it in its:
        by_key[it["group_key"]].append(it)
    out = []
    for k in sorted(by_key):
        ks = sorted(by_key[k], key=lambda x: x["img"])
        step = len(ks) / n
        out.extend(ks if len(ks) <= n else [ks[int(i * step)] for i in range(n)])
    return out


def split_groups(items, ratios, seed, train_cap=None):
    by_group = defaultdict(list)
    for it in items:
        by_group[it["group"]].append(it)
    # Stratify by each group's dominant class so rare classes reach val/test.
    by_cls = defaultdict(list)
    for g, its in by_group.items():
        dom = Counter(b[0] for it in its for b in it["boxes"]).most_common(1)[0][0]
        by_cls[dom].append(g)
    rng = random.Random(seed)
    assign = {}
    for cls in sorted(by_cls):
        gs = sorted(by_cls[cls])
        rng.shuffle(gs)
        n_val = max(1, round(len(gs) * ratios["val"]))
        n_test = max(1, round(len(gs) * ratios["test"]))
        for k, g in enumerate(gs):
            assign[g] = "val" if k < n_val else "test" if k < n_val + n_test else "train"
    out = defaultdict(list)
    for g, its in by_group.items():
        split = assign[g]
        out[split].extend(cap_per_key(its, train_cap) if split == "train" else its[:1])
    return out


def write(cfg, splits):
    out_dir = cfg["out_dir"]
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    classes = cfg["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    stats = {}
    for split, its in splits.items():
        os.makedirs(os.path.join(out_dir, split, "images"))
        os.makedirs(os.path.join(out_dir, split, "labels"))
        boxes, sources = Counter(), Counter()
        for n, it in enumerate(sorted(its, key=lambda x: x["img"])):
            name = f"{it['source'].replace('/', '_')}_{n:06d}"
            ext = os.path.splitext(it["img"])[1].lower()
            dst = os.path.join(out_dir, split, "images", name + ext)
            try:
                os.link(it["img"], dst)
            except OSError:
                shutil.copy2(it["img"], dst)
            with open(os.path.join(out_dir, split, "labels", name + ".txt"), "w") as f:
                for c, cx, cy, w, h in it["boxes"]:
                    f.write(f"{cidx[c]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    boxes[c] += 1
            sources[it["source"]] += 1
        stats[split] = {"images": len(its), "boxes": dict(sorted(boxes.items())),
                        "sources": dict(sorted(sources.items()))}
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": ".", "train": "train/images", "val": "val/images",
                        "test": "test/images", "names": dict(enumerate(classes))},
                       f, sort_keys=False)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    items = collect(cfg)
    print(f"collected {len(items)} labelled images")
    cache = os.path.join(cfg["raw_dir"], ".dhash_cache.json")
    items = build_groups(items, cfg.get("dhash_max_dist", 8), cache)
    n_groups = len({it["group"] for it in items})
    print(f"{len({it['group_key'] for it in items})} originals -> {n_groups} groups after dedupe")
    splits = split_groups(items, cfg["split"], cfg.get("seed", 0), cfg.get("train_max_per_key"))
    stats = write(cfg, splits)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
