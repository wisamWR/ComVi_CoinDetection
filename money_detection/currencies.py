"""Currency definitions: model class name -> display label and face value."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class Currency:
    code: str
    name: str
    weights: Path
    # class name -> (display label, value in minor units, e.g. cents)
    denoms: dict
    minor_per_major: int = 1

    def value(self, cls):
        return self.denoms[cls][1]

    def label(self, cls):
        return self.denoms[cls][0]

    def format(self, minor):
        if self.code == "usd":
            return f"${minor / self.minor_per_major:,.2f}"
        return f"Rp {minor:,}".replace(",", ".")

    def total(self, classes):
        return sum(self.value(c) for c in classes)

    def breakdown(self, classes):
        """Rows of (label, count, subtotal), highest denomination first."""
        counts = Counter(classes)
        order = sorted(counts, key=self.value, reverse=True)
        return [(self.label(c), counts[c], counts[c] * self.value(c)) for c in order]


def _idr(*values):
    return {f"idr_{v}": (f"Rp {v:,}".replace(",", "."), v) for v in values}


CURRENCIES = {
    "idr": Currency(
        code="idr",
        name="Rupiah (uang kertas)",
        weights=MODELS_DIR / "idr_notes.pt",
        denoms=_idr(1000, 2000, 5000, 10000, 20000, 50000, 100000),
    ),
    "usd": Currency(
        code="usd",
        name="US Dollar (koin)",
        weights=MODELS_DIR / "usd_coins.pt",
        denoms={
            "usd_penny": ("Penny", 1),
            "usd_nickel": ("Nickel", 5),
            "usd_dime": ("Dime", 10),
            "usd_quarter": ("Quarter", 25),
        },
        minor_per_major=100,
    ),
}
