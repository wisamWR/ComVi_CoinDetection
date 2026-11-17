"""YOLO inference wrapper: frame -> detections with denomination values, plus drawing."""
from dataclasses import dataclass

import cv2

from .currencies import CURRENCIES

# Tableau 10, BGR
COLORS = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
          (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple
    cls_id: int


class MoneyDetector:
    def __init__(self, currency="idr", weights=None, device=None):
        from ultralytics import YOLO  # heavy import, keep it out of module import time

        self.currency = CURRENCIES[currency]
        self.model = YOLO(str(weights or self.currency.weights), task="detect")
        self.device = device
        unknown = set(self.model.names.values()) - set(self.currency.denoms)
        if unknown:
            raise ValueError(f"Model classes {sorted(unknown)} are not {self.currency.code} denominations")

    def detect(self, frame, conf=0.5):
        r = self.model(frame, conf=conf, device=self.device, verbose=False)[0]
        return [
            Detection(r.names[int(k)], float(c), tuple(int(v) for v in box), int(k))
            for box, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())
        ]


def annotate(frame, detections, currency):
    """Draw boxes, labels and a running total on a copy of a BGR frame."""
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        color = COLORS[d.cls_id % len(COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{currency.label(d.cls)} {d.conf:.0%}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty = max(y1, th + 10)
        cv2.rectangle(out, (x1, ty - th - 10), (x1 + tw, ty + base - 10), color, cv2.FILLED)
        cv2.putText(out, text, (x1, ty - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    total = currency.format(currency.total(d.cls for d in detections))
    banner = f"Total: {total} ({len(detections)} item)"
    cv2.putText(out, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out
