"""Model-free tests: text formatting, reading order, the non-egypt rule, page order, JSON round trip, relayout."""
from akhet import Page, Line, Sign, NE_TOKEN, text_blocks
from akhet.layout import order_line, order_lines, remove_signs_inside_non_egypt
from akhet.text import tla_code


def test_tla_code():
    assert tla_code("AA1") == "Aa1" and tla_code("DAMAGED") == "//" and tla_code("G17") == "G17"


def _row(direction, ne=()):
    signs = [{"code": "A1", "x1": 0, "y1": 0, "x2": 10, "y2": 10}, {"code": "G17", "x1": 40, "y1": 0, "x2": 50, "y2": 10},
             {"code": "N35", "x1": 60, "y1": 0, "x2": 70, "y2": 10}]
    return order_line(signs, True, direction, (0, 0, 120, 10), ne)


def test_reading_direction_and_ne_token():
    assert _row("Left")["codes"] == ["A1", "G17", "N35"]
    assert _row("Right")["codes"] == ["N35", "G17", "A1"]
    assert _row("Unknown")["direction_used"] == "Left"
    assert _row("Left", [(15, 0, 35, 10)])["tokens"] == ["A1", NE_TOKEN, "G17", "N35"]


def test_non_egypt_wins_only_on_real_overlap():
    keep, removed = remove_signs_inside_non_egypt([(0, 0, 10, 10), (20, 0, 30, 10)], [(8, 0, 40, 10)], thr=0.5)
    assert keep == [0] and removed == [1]           # sign 0 touches 20 %, sign 1 lies fully inside


def test_page_order_columns_right_to_left():
    cols = [{"box": [x, 0, x + 50, 600], "is_row": False, "direction": "Right"} for x in (0, 60, 120)]
    assert order_lines(cols)[0] == [2, 1, 0]


def test_page_roundtrip_relayout_and_blocks(tmp_path):
    ln = Line(id=1, box=[0, 0, 120, 10], score=0.9, is_row=True, direction="Left", direction_score=0.8,
              signs=[Sign([60, 0, 70, 10], 0.9, "N35", 0.8, [("N35", 0.8)]), Sign([0, 0, 10, 10], 0.9, "AA1", 0.7, [("AA1", 0.7)])])
    page = Page(width=120, height=10, lines=[ln]).relayout()
    assert page.lines[0].codes == ["AA1", "N35"]
    page.to_json(tmp_path / "p.json")
    back = Page.from_json(tmp_path / "p.json")
    back.lines[0].direction = "Right"
    assert back.relayout().lines[0].codes == ["N35", "AA1"]
    assert text_blocks(back)[0].hieroglyphs == "N35 Aa1"
