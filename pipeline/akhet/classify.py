"""Sign classifier: metric-learning embedder (DINOv2-base, ONNX) + prototype bank, cosine nearest prototype.

Crop view = the training eval view: expand the box by CTX (0.075) of its size per side, clamp to the page, crop, then
`present`: scale the longest side to S*(1-2*0.03) (LANCZOS when shrinking, BICUBIC when growing), paste centered on an
S x S canvas filled with the crop's median border colour, ImageNet mean/std.
Prototype bank B: per class normalize(normalize(mean real emb) + normalize(mean synth emb)), one vector per class.
"""
import json
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _median_border(im):
    a = np.asarray(im)
    ring = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(ring, 0))


def present(im, S=224, margin=0.03):
    im = im.convert("RGB"); w, h = im.size
    sc = S * (1 - 2 * margin) / max(w, h); nw, nh = max(1, round(w * sc)), max(1, round(h * sc))
    c = Image.new("RGB", (S, S), _median_border(im))
    c.paste(im.resize((nw, nh), Image.LANCZOS if sc < 1 else Image.BICUBIC), ((S - nw) // 2, (S - nh) // 2))
    return c


def crop_view(page, box, ctx=0.075, S=224):
    """page: PIL RGB image, box: (x1, y1, x2, y2) integer pixels -> normalized CHW float32."""
    x1, y1, x2, y2 = box
    if ctx > 0:
        bw, bh = x2 - x1, y2 - y1; W, H = page.size
        x1, y1 = max(0, int(round(x1 - ctx * bw))), max(0, int(round(y1 - ctx * bh)))
        x2, y2 = min(W, int(round(x2 + ctx * bw))), min(H, int(round(y2 + ctx * bh)))
    im = present(page.crop((x1, y1, x2, y2)).convert("RGB"), S)
    return ((np.asarray(im, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


class SignClassifier:
    def __init__(self, onnx_path, bank_npy, bank_json, providers, threads=None, batch=64):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(onnx_path, so, providers=list(providers))
        self.P = np.load(bank_npy).astype(np.float32)
        meta = json.load(open(bank_json))
        self.codes = list(meta["codes"]); self.ctx = float(meta.get("ctx_margin", 0.075)); self.S = int(meta.get("S", 224))
        self.batch = batch

    def embed(self, x):
        return np.concatenate([self.sess.run(None, {"image": np.ascontiguousarray(x[i:i + self.batch])})[0]
                               for i in range(0, len(x), self.batch)])

    def classify(self, page, boxes, topk=5):
        """boxes: list of (x1, y1, x2, y2) floats on the page -> list of [(code, cosine), ...] top-k per box."""
        if not boxes:
            return []
        ib = []
        for b in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in b)
            ib.append((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
        x = np.stack([crop_view(page, b, self.ctx, self.S) for b in ib]).astype(np.float32)
        sim = self.embed(x) @ self.P.T
        top = np.argsort(-sim, 1)[:, :topk]
        return [[(self.codes[j], float(sim[i, j])) for j in row] for i, row in enumerate(top)]
