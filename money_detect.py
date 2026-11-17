"""Detect money in images, video or a live camera and show the total value.

Examples:
  python money_detect.py --currency idr --source usb0 --resolution 1280x720
  python money_detect.py --currency usd --source coins.jpg
  python money_detect.py --currency idr --source ./photos/ --thresh 0.6

Keys: Q quit, S pause, P save capture.png
"""
import argparse
import glob
import os
import sys
import time
from collections import deque

import cv2

from money_detection import CURRENCIES, MoneyDetector, annotate

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXT = {".avi", ".mov", ".mp4", ".mkv", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--currency", choices=CURRENCIES, required=True)
    p.add_argument("--source", required=True,
                   help='image file, image folder, video file, USB camera ("usb0") or Picamera ("picamera0")')
    p.add_argument("--model", help="override weights path (default: models/<currency model>.pt)")
    p.add_argument("--thresh", type=float, default=0.5, help="minimum confidence (default 0.5)")
    p.add_argument("--resolution", help='display resolution WxH, e.g. "640x480"')
    p.add_argument("--record", action="store_true", help="record video/camera output to demo1.avi")
    p.add_argument("--device", help='inference device, e.g. "cpu" or "0"')
    return p.parse_args()


def source_type(src):
    if os.path.isdir(src):
        return "folder"
    if os.path.isfile(src):
        ext = os.path.splitext(src)[1].lower()
        if ext in IMG_EXT:
            return "image"
        if ext in VID_EXT:
            return "video"
        sys.exit(f"File extension {ext} is not supported.")
    if src.startswith("usb"):
        return "usb"
    if src.startswith("picamera"):
        return "picamera"
    sys.exit(f"Input {src} is invalid.")


def frames(src, kind, res):
    """Yield BGR frames from the source."""
    if kind in ("image", "folder"):
        paths = [src] if kind == "image" else sorted(
            f for f in glob.glob(os.path.join(src, "*")) if os.path.splitext(f)[1].lower() in IMG_EXT)
        for path in paths:
            yield cv2.imread(path)
        return
    if kind == "picamera":
        from picamera2 import Picamera2
        cam = Picamera2()
        size = res or (1280, 720)
        cam.configure(cam.create_video_configuration(main={"format": "XRGB8888", "size": size}))
        cam.start()
        try:
            while True:
                yield cv2.cvtColor(cam.capture_array(), cv2.COLOR_BGRA2BGR)
        finally:
            cam.stop()
    cap = cv2.VideoCapture(src if kind == "video" else int(src[3:]))
    if res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("End of video or camera disconnected.")
                return
            yield frame
    finally:
        cap.release()


def main():
    args = parse_args()
    kind = source_type(args.source)
    res = tuple(int(v) for v in args.resolution.split("x")) if args.resolution else None
    live = kind in ("video", "usb", "picamera")

    recorder = None
    if args.record:
        if not live:
            sys.exit("Recording only works for video and camera sources.")
        if not res:
            sys.exit("Please specify --resolution to record.")
        recorder = cv2.VideoWriter("demo1.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, res)

    detector = MoneyDetector(args.currency, weights=args.model, device=args.device)
    currency = detector.currency
    fps = deque(maxlen=200)

    for frame in frames(args.source, kind, res):
        t0 = time.perf_counter()
        if res:
            frame = cv2.resize(frame, res)
        dets = detector.detect(frame, conf=args.thresh)
        out = annotate(frame, dets, currency)
        if live and fps:
            cv2.putText(out, f"FPS: {sum(fps) / len(fps):.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if not live:
            print(f"Total: {currency.format(currency.total(d.cls for d in dets))}",
                  {label: n for label, n, _ in currency.breakdown(d.cls for d in dets)})

        cv2.imshow("ComVi money detection", out)
        if recorder:
            recorder.write(out)

        key = cv2.waitKey(5 if live else 0) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            cv2.waitKey()
        elif key in (ord("p"), ord("P")):
            cv2.imwrite("capture.png", out)
        fps.append(1 / (time.perf_counter() - t0))

    if fps:
        print(f"Average pipeline FPS: {sum(fps) / len(fps):.2f}")
    if recorder:
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
