"""Download Roboflow dataset versions (YOLO format) into <out>/<slug>/."""
import json
import os
import sys
import time
import urllib.request
import zipfile

def _load_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key and os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            key = winreg.QueryValueEx(k, "ROBOFLOW_API_KEY")[0]
    if not key:
        sys.exit("ROBOFLOW_API_KEY not set (run: setx ROBOFLOW_API_KEY <key>)")
    return key


KEY = _load_key()
OUT = sys.argv[1]
TARGETS = [
    ("dimas-q1ykr/rupiah-emisi-2022-xqjcj", 1),
    ("indonesia-rupiah-currency-dataset/indonesia-rupiah-detection", 3),
    ("4ia17ottos-workspace/uang-rupiah-baru", 4),
    ("project-lca4e/uang-rupiah-oyrxk-qkrud", 3),
    ("coin-new/uang-rupiah-kertas-dan-koin-by-adelia-clarissa-zzt2j", 11),
]


def download_resumable(url, path, retries=50):
    """Stream url to path; on stall/disconnect resume with an HTTP Range request."""
    for attempt in range(retries):
        done = os.path.getsize(path) if os.path.exists(path) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if done and r.status != 206:  # server ignored Range: start over
                    done = 0
                total = done + int(r.headers.get("Content-Length", 0))
                with open(path, "ab" if done else "wb") as f:
                    last = time.time()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            if total > done:
                                raise OSError(f"stream ended early at {done} bytes")
                            return
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 10:
                            print(f"   {done / 1e6:.0f}/{total / 1e6:.0f} MB")
                            last = time.time()
        except (OSError, TimeoutError) as e:
            print(f"   retry {attempt + 1} after error: {e}")
            time.sleep(5)
    raise RuntimeError("download failed after retries: " + path)


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


for proj, ver in TARGETS:
    slug = proj.split("/")[1]
    dest = os.path.join(OUT, slug)
    if os.path.isdir(dest) and os.listdir(dest):
        print("skip (exists)", slug)
        continue
    url = f"https://api.roboflow.com/{proj}/{ver}/yolov8?api_key={KEY}"
    for attempt in range(30):
        d = get_json(url)
        link = (d.get("export") or {}).get("link")
        if link:
            break
        print("  waiting for export...", slug, d.get("progress"))
        time.sleep(10)
    else:
        print("FAILED export", slug)
        continue
    print("downloading", slug, "v", ver)
    zpath = dest + ".zip.part"
    download_resumable(link, zpath)
    print("  ", round(os.path.getsize(zpath) / 1e6, 1), "MB")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            target = os.path.realpath(os.path.join(dest, m))
            if not target.startswith(os.path.realpath(dest)):
                raise RuntimeError("unsafe path in zip: " + m)
        z.extractall(dest)
    os.remove(zpath)
print("done")
