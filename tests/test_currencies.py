import pytest

from money_detection import CURRENCIES

IDR = CURRENCIES["idr"]
USD = CURRENCIES["usd"]


def test_classes_match_dataset_configs():
    assert list(IDR.denoms) == ["idr_1000", "idr_2000", "idr_5000", "idr_10000",
                                "idr_20000", "idr_50000", "idr_100000"]
    assert list(USD.denoms) == ["usd_penny", "usd_nickel", "usd_dime", "usd_quarter"]


def test_weights_point_into_models_dir():
    assert IDR.weights.name == "idr_notes.pt"
    assert USD.weights.name == "usd_coins.pt"
    assert IDR.weights.parent.name == "models"


@pytest.mark.parametrize("minor, text", [(0, "Rp 0"), (1000, "Rp 1.000"), (175000, "Rp 175.000"),
                                         (1250000, "Rp 1.250.000")])
def test_idr_format(minor, text):
    assert IDR.format(minor) == text


@pytest.mark.parametrize("minor, text", [(0, "$0.00"), (41, "$0.41"), (125, "$1.25"), (123456, "$1,234.56")])
def test_usd_format(minor, text):
    assert USD.format(minor) == text


def test_totals():
    assert IDR.total(["idr_100000", "idr_50000", "idr_2000", "idr_2000"]) == 154000
    assert USD.total(["usd_quarter", "usd_dime", "usd_nickel", "usd_penny", "usd_penny"]) == 42
    assert IDR.total([]) == 0


def test_breakdown_sorted_by_value():
    rows = USD.breakdown(["usd_penny", "usd_quarter", "usd_penny", "usd_dime"])
    assert rows == [("Quarter", 1, 25), ("Dime", 1, 10), ("Penny", 2, 2)]


def test_labels():
    assert IDR.label("idr_20000") == "Rp 20.000"
    assert USD.label("usd_nickel") == "Nickel"


def test_unknown_class_raises():
    with pytest.raises(KeyError):
        IDR.total(["usd_penny"])
