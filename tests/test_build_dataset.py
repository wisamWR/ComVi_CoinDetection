import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from build_dataset import drop_nested  # noqa: E402

NOTE = ("idr_5000", 0.50, 0.50, 0.90, 0.80)
LOGO = ("idr_5000", 0.20, 0.20, 0.10, 0.10)   # inside NOTE
PORTRAIT = ("idr_5000", 0.65, 0.45, 0.40, 0.70)  # inside NOTE


def test_drops_same_class_parts_inside_whole_note():
    assert drop_nested([LOGO, NOTE, PORTRAIT]) == [NOTE]


def test_keeps_box_inside_box_of_other_class():
    small = ("idr_2000", *LOGO[1:])
    assert drop_nested([NOTE, small]) == [NOTE, small]


def test_keeps_side_by_side_notes():
    left = ("idr_5000", 0.25, 0.5, 0.4, 0.6)
    right = ("idr_5000", 0.75, 0.5, 0.4, 0.6)
    assert drop_nested([left, right]) == [left, right]


def test_keeps_one_of_identical_duplicates():
    assert drop_nested([NOTE, NOTE]) == [NOTE]


def test_tolerates_box_poking_slightly_out():
    edge = ("idr_5000", 0.50, 0.50, 0.905, 0.40)  # 0.0025 wider than NOTE on each side
    assert drop_nested([NOTE, edge]) == [NOTE]
