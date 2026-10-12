"""akhet OCR result objects: Page -> Line -> Sign, JSON export and a drawing helper."""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from .layout import NE_TOKEN


@dataclass
class Sign:
    box: List[float]                      # x1, y1, x2, y2 on the page
    score: float                          # detector confidence
    code: str                             # Gardiner code (top-1)
    similarity: float                     # cosine to the top-1 prototype
    candidates: List[Tuple[str, float]]   # top-k (code, cosine)


@dataclass
class Line:
    id: int
    box: List[int]
    score: float
    is_row: bool
    direction: str                        # Left | Right | Unknown (predicted)
    direction_score: float
    signs: List[Sign] = field(default_factory=list)          # in READING ORDER
    non_egypt: List[dict] = field(default_factory=list)      # {"box", "score"}
    removed_signs: List[dict] = field(default_factory=list)  # hieroglyph boxes removed by the non-egypt rule
    tokens: List[str] = field(default_factory=list)          # codes in reading order, NE_TOKEN at non-egypt positions
    mdc: str = ""                                             # packed Manuel de Codage
    n_quadrats: int = 0
    order_confidence: float = 0.0
    reading_direction: str = "Left"                          # direction used for ordering (Unknown -> Left)
    page_index: int = 0                                       # position in the page reading order
    block: int = 0                                            # text block id (blocks in reading order)

    @property
    def codes(self):
        return [s.code for s in self.signs]

    @property
    def gardiner(self):
        """space-separated Gardiner text of the line, non-egypt dropped"""
        return " ".join(t for t in self.tokens if t != NE_TOKEN)


@dataclass
class Page:
    width: int
    height: int
    lines: List[Line] = field(default_factory=list)          # sorted by page_index
    source: Optional[str] = None

    def blocks(self):
        """[[Line, ...], ...] text blocks in reading order (the unit for transliteration / translation)."""
        out = {}
        for ln in self.lines:
            out.setdefault(ln.block, []).append(ln)
        return [out[k] for k in sorted(out)]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        lines = []
        for l in d["lines"]:
            l = dict(l)
            l["signs"] = [Sign(**{**s, "candidates": [tuple(c) for c in s.get("candidates", [])]}) for s in l.get("signs", [])]
            lines.append(Line(**l))
        return cls(width=d["width"], height=d["height"], lines=lines, source=d.get("source"))

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.load(open(path, encoding="utf-8")))

    def relayout(self):
        """Recompute reading order, tokens, MdC, page order and blocks from the CURRENT signs and line fields.
        Call it after hand edits: changed codes or boxes, added / removed signs (Sign(box, score, code, ...)),
        a corrected line.direction ('Left' / 'Right') or line.is_row, edited non_egypt boxes.
        Ordering is geometric (quadrats from the boxes). To impose a different sign order, edit the text blocks."""
        from .layout import order_line, order_lines
        for ln in self.lines:
            r = order_line([{"code": s.code, "x1": s.box[0], "y1": s.box[1], "x2": s.box[2], "y2": s.box[3]} for s in ln.signs],
                           ln.is_row, ln.direction, ln.box, [n["box"] for n in ln.non_egypt])
            ln.signs = [ln.signs[i] for i in r["order"]]
            ln.tokens, ln.mdc, ln.n_quadrats = r["tokens"], r["packed"], r["n_quads"]
            ln.order_confidence, ln.reading_direction = float(r["conf"]), r["direction_used"]
        order, block = order_lines([{"box": ln.box, "is_row": ln.is_row, "direction": ln.direction} for ln in self.lines])
        for rank, i in enumerate(order):
            self.lines[i].page_index = rank; self.lines[i].block = block[i]
        self.lines.sort(key=lambda l: l.page_index)
        return self

    def to_json(self, path=None, indent=1):
        s = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            open(path, "w", encoding="utf-8").write(s)
        return s

    def draw(self, image, max_side=2600, font_size=14):
        """blue = line (reading index, block, row/col, direction), green = hieroglyph 'index:code' in reading order,
        red = non-egypt, orange dashed = hieroglyph removed by the non-egypt rule."""
        from PIL import ImageDraw, ImageFont
        from .io import load_image
        im = load_image(image).copy(); dr = ImageDraw.Draw(im)
        try:
            f = ImageFont.truetype("DejaVuSans.ttf", font_size); fl = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size + 8)
        except OSError:
            f = fl = ImageFont.load_default()
        for ln in self.lines:
            x1, y1, x2, y2 = ln.box
            dr.rectangle([x1, y1, x2, y2], outline=(40, 90, 255), width=3)
            dr.text((x1 + 2, max(0, y1 - font_size - 10)),
                    f"#{ln.page_index + 1} b{ln.block} {'Row' if ln.is_row else 'Col'} {ln.direction} ({ln.direction_score:.2f})",
                    fill=(40, 90, 255), font=fl)
            for k, s in enumerate(ln.signs):
                dr.rectangle(s.box, outline=(0, 170, 0), width=2)
                dr.text((s.box[0] + 1, s.box[1] + 1), f"{k + 1}:{s.code}", fill=(0, 120, 0), font=f)
            for ne in ln.non_egypt:
                dr.rectangle(ne["box"], outline=(230, 0, 0), width=3)
            for g in ln.removed_signs:
                a, b, c, d = g["box"]
                for xa in range(int(a), int(c), 12):
                    dr.line([xa, b, min(xa + 6, c), b], fill=(255, 140, 0), width=3); dr.line([xa, d, min(xa + 6, c), d], fill=(255, 140, 0), width=3)
                for ya in range(int(b), int(d), 12):
                    dr.line([a, ya, a, min(ya + 6, d)], fill=(255, 140, 0), width=3); dr.line([c, ya, c, min(ya + 6, d)], fill=(255, 140, 0), width=3)
        s = min(1.0, max_side / max(im.size))
        return im.resize((round(im.size[0] * s), round(im.size[1] * s))) if s < 1 else im
