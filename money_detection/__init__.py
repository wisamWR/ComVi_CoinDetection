from .currencies import CURRENCIES, Currency

__all__ = ["CURRENCIES", "Currency", "MoneyDetector", "Detection", "annotate"]


def __getattr__(name):
    # Lazy so that importing currencies (e.g. in tests) does not pull in cv2/ultralytics.
    if name in ("MoneyDetector", "Detection", "annotate"):
        from . import detector
        return getattr(detector, name)
    raise AttributeError(name)
