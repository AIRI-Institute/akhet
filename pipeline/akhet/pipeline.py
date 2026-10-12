"""Akhet OCR pipeline: page image -> lines -> hieroglyph boxes -> non-egypt + direction -> sign codes -> reading order.

  1 lines        line detector, square 1024, keep score >= line_thr. Row if wider than tall.
  2 hieroglyphs  tiled hieroglyph detector (seqA) on the raw line crop, short side 256 (LANCZOS).
  3 non-egypt    direction + non-egypt detector, square 1024, on the line crop + 6 px, short side 256.
    + direction  Line direction = class of its top-scoring direction box (none -> Unknown). Non-egypt kept at
                 score >= nonegypt_thr. This detector's own hieroglyph class is not used.
  4 conflict     non-egypt wins: a hieroglyph with >= conflict_thr of its area inside a non-egypt box is removed.
  5 signs        metric classifier + prototype bank, top-k codes with cosine similarity.
  6 order        quadrat grouping + reading order + packed MdC per line, then the page line order and text blocks.
"""
import json, os
import numpy as np
from PIL import Image

from .detection import Detector, seqa
from .classify import SignClassifier
from .layout import remove_signs_inside_non_egypt, order_line, order_lines
from .result import Page, Line, Sign
from .io import load_image

DIRECTIONS = {2: "Left", 3: "Right", 4: "Unknown"}   # label indices of the direction + non-egypt detector
NON_EGYPT = 1

DEFAULTS = dict(line_thr=0.5, nonegypt_thr=0.5, conflict_thr=0.5, topk=5, crop_short=256, nonegypt_pad=6)


def _providers(device):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if device == "cpu" or (device == "auto" and "CUDAExecutionProvider" not in avail):
        return ("CPUExecutionProvider",)
    try:
        ort.preload_dlls()          # CUDA 12 / cuDNN 9 from the onnxruntime-gpu[cuda,cudnn] wheels, if installed
    except Exception:
        pass
    return ("CUDAExecutionProvider", "CPUExecutionProvider")


HUB_REPO = "mxmgolyadkin/akhet"     # default weights repo on the Hugging Face Hub


def _resolve(repo_or_dir, patterns):
    repo_or_dir = repo_or_dir or HUB_REPO
    if os.path.isdir(repo_or_dir):
        return repo_or_dir
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_or_dir, allow_patterns=patterns)


def _crop_short(page, box, pad, short):
    x1, y1, x2, y2 = box; W, H = page.size
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    c = page.crop((x1, y1, x2, y2)); s = short / float(min(c.size))
    c = c.resize((max(1, round(c.size[0] * s)), max(1, round(c.size[1] * s))), Image.LANCZOS)
    return np.asarray(c), (x1, y1, c.size[0] / (x2 - x1), c.size[1] / (y2 - y1))


def _to_page(b, m):
    ox, oy, sx, sy = m
    return [b[0] / sx + ox, b[1] / sy + oy, b[2] / sx + ox, b[3] / sy + oy]


class Akhet:
    """ocr = Akhet("models/"); page = ocr("page.jpg"); page.lines[0].mdc"""

    FILES = dict(lines="lines.onnx", glyphs_row="glyphs_row.onnx", glyphs_col="glyphs_col.onnx",
                 direction_nonegypt="direction_nonegypt.onnx", classifier="sign_classifier.onnx",
                 bank="sign_bank.npy", bank_meta="sign_bank.json")

    @classmethod
    def from_pretrained(cls, repo_or_dir=None, **kw):
        """Load from a local model folder or download the ONNX models from the Hugging Face Hub (needs huggingface_hub)."""
        return cls(_resolve(repo_or_dir, ["*.onnx", "sign_bank.npy", "sign_bank.json"]), **kw)

    def __init__(self, model_dir, device="auto", threads=None, **cfg):
        self.cfg = dict(DEFAULTS); self.cfg.update(cfg)
        p = lambda k: os.path.join(model_dir, self.FILES[k])
        prov = _providers(device)
        self.line_det = Detector(p("lines"), "square", providers=prov, threads=threads)
        self.glyph_det = Detector({"row": p("glyphs_row"), "col": p("glyphs_col")}, "thick", providers=prov, threads=threads)
        self.dirne_det = Detector(p("direction_nonegypt"), "square", providers=prov, threads=threads)
        self.classifier = SignClassifier(p("classifier"), p("bank"), p("bank_meta"), providers=prov, threads=threads)

    def __call__(self, image, direction=None):
        """image: path | PIL | ndarray. direction: None = predicted per line, or force 'Left' / 'Right' for every line."""
        c = self.cfg
        page = load_image(image); W, H = page.size; arr = np.asarray(page)
        res = Page(width=W, height=H, source=image if isinstance(image, str) else None)

        # 1 lines
        for lab, sc, x1, y1, x2, y2 in self.line_det(arr):
            if sc < c["line_thr"]:
                continue
            bx = [max(0, int(round(x1))), max(0, int(round(y1))), min(W, int(round(x2))), min(H, int(round(y2)))]
            if bx[2] - bx[0] < 4 or bx[3] - bx[1] < 4:
                continue
            res.lines.append(Line(id=len(res.lines) + 1, box=bx, score=float(sc), is_row=(bx[2] - bx[0]) >= (bx[3] - bx[1]),
                                  direction="Unknown", direction_score=0.0))

        boxes_per_line = []
        for ln in res.lines:
            # 2 hieroglyphs
            a1, m1 = _crop_short(page, ln.box, 0, c["crop_short"])
            gl = [(_to_page(b, m1), sc) for sc, b, cl in seqa(a1, self.glyph_det)]
            # 3 non-egypt + direction
            a5, m5 = _crop_short(page, ln.box, c["nonegypt_pad"], c["crop_short"])
            dirs = []
            for lab, sc, x1, y1, x2, y2 in self.dirne_det(a5):
                lab = int(lab)
                if lab in DIRECTIONS:
                    dirs.append((float(sc), DIRECTIONS[lab]))
                elif lab == NON_EGYPT and sc >= c["nonegypt_thr"]:
                    ln.non_egypt.append({"box": _to_page([x1, y1, x2, y2], m5), "score": float(sc)})
            ln.direction_score, ln.direction = max(dirs, default=(0.0, "Unknown"))
            # 4 conflict: non-egypt wins
            keep, removed = remove_signs_inside_non_egypt([g[0] for g in gl], [n["box"] for n in ln.non_egypt], c["conflict_thr"])
            ln.removed_signs = [{"box": gl[i][0], "score": float(gl[i][1])} for i in removed]
            boxes_per_line.append([gl[i] for i in keep])

        # 5 signs (one batch per page)
        flat = [g for gl in boxes_per_line for g in gl]
        cands = self.classifier.classify(page, [g[0] for g in flat], topk=c["topk"])
        k = 0
        for ln, gl in zip(res.lines, boxes_per_line):
            signs = []
            for b, sc in gl:
                cd = cands[k]; k += 1
                signs.append(Sign(box=[float(v) for v in b], score=float(sc), code=cd[0][0], similarity=cd[0][1], candidates=cd))
            # 6 reading order inside the line
            d = direction or ln.direction
            r = order_line([{"code": s.code, "x1": s.box[0], "y1": s.box[1], "x2": s.box[2], "y2": s.box[3]} for s in signs],
                           ln.is_row, d, ln.box, [n["box"] for n in ln.non_egypt])
            ln.signs = [signs[i] for i in r["order"]]
            ln.tokens, ln.mdc, ln.n_quadrats = r["tokens"], r["packed"], r["n_quads"]
            ln.order_confidence, ln.reading_direction = float(r["conf"]), r["direction_used"]

        # 6 page line order + blocks
        order, block = order_lines([{"box": ln.box, "is_row": ln.is_row, "direction": direction or ln.direction} for ln in res.lines])
        for rank, i in enumerate(order):
            res.lines[i].page_index = rank; res.lines[i].block = block[i]
        res.lines.sort(key=lambda l: l.page_index)
        return res
