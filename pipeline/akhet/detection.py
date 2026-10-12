"""akhet detectors: numpy pre/post-processing around the ONNX RT-DETR graphs (exported WITHOUT ppdet post-process).
Reproduces PaddleDetection inference exactly:
  square  (lines, dirne): Decode RGB -> Resize keep_ratio long side 1024 (int(s*x+0.5), cv2 INTER_CUBIC, fx/fy call)
                          -> Pad 1024 top-left fill (123.675, 116.28, 103.53) -> /255 -> CHW. Decode = 'pad' with the
                          pad_shape = max(im_shape) fix (eval_ppdet_regime.py).
  thick   (glyphs):       ThickCropPadFixed eval: downscale-to-fit long <= 2048 (INTER_LINEAR), paste top-left on a
                          fixed 256 x 2048 (row) / 2048 x 256 (col) canvas of 127 -> /255 -> CHW. Decode = 'origin'.
  both:                   sigmoid(logits), top-300 over (query x class) flattened, label = idx % C.
Returns per image an array [N, 6] = (label, score, x1, y1, x2, y2) in ORIGINAL image pixels, sorted by score desc.
"""
import numpy as np
import cv2

PAD_FILL = np.array([123.675, 116.28, 103.53], dtype=np.float32)
TOPK = 300


def prep_square(img, size=1024):
    h, w = img.shape[:2]
    s = min(size / min(h, w), size / max(h, w))
    rh, rw = int(s * h + 0.5), int(s * w + 0.5)
    im = cv2.resize(img, None, None, fx=rw / w, fy=rh / h, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    canvas = np.ones((size, size, 3), np.float32) * PAD_FILL
    canvas[:rh, :rw] = im
    x = (canvas / 255.0).transpose(2, 0, 1)
    pad = float(max(rh, rw))
    out_wh = (pad / rw * w, pad / rh * h)            # 'pad' decode with the max(im_shape) pad fix
    return x, out_wh


def prep_thick(img, short=256, max_long=2048, fill=127):
    H, W = img.shape[:2]
    row = H <= W
    long_ = W if row else H
    scale = min(1.0, max_long / float(long_))
    if scale < 1.0:
        W2, H2 = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        img = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_LINEAR)
        H, W = H2, W2
    canvas = np.full((short, max_long, 3) if row else (max_long, short, 3), fill, dtype=img.dtype)
    canvas[:H, :W] = img
    ch, cw = canvas.shape[:2]
    x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)
    out_wh = (np.floor(cw / scale + 0.5), np.floor(ch / scale + 0.5))   # 'origin' decode
    return x, out_wh


def decode(boxes, logits, out_wh, topk=TOPK):
    """boxes [Q,4] cxcywh normalized, logits [Q,C] -> [K,6] (label, score, x1,y1,x2,y2), score-sorted."""
    C = logits.shape[1]
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    flat = scores.reshape(-1)
    k = min(topk, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx], kind="stable")]
    q, lab = idx // C, idx % C
    cx, cy, bw, bh = boxes[q, 0], boxes[q, 1], boxes[q, 2], boxes[q, 3]
    W, H = out_wh
    xyxy = np.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], 1)
    return np.concatenate([lab[:, None].astype(np.float64), flat[idx][:, None], xyxy], 1)


class Detector:
    """mode "square": one ONNX path. mode "thick": dict {"row": path [1,3,256,2048], "col": path [1,3,2048,256]}
    (static shapes, the dynamic-H/W export breaks in onnxruntime)."""
    def __init__(self, onnx_path, mode, providers=("CUDAExecutionProvider", "CPUExecutionProvider"), threads=None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError('akhet needs an ONNX runtime: pip install "akhet[cpu]" or "akhet[gpu]"') from e
        self.mode = mode
        paths = onnx_path if isinstance(onnx_path, dict) else {"one": onnx_path}
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = {k: ort.InferenceSession(p, so, providers=list(providers)) for k, p in paths.items()}
        self.inp = {k: s.get_inputs()[0].name for k, s in self.sess.items()}

    def __call__(self, img_rgb):
        if self.mode == "square":
            x, out_wh = prep_square(img_rgb); k = "one"
        else:
            x, out_wh = prep_thick(img_rgb); k = "row" if x.shape[1] < x.shape[2] else "col"
        boxes, logits = self.sess[k].run(None, {self.inp[k]: x[None].astype(np.float32)})
        return decode(boxes[0], logits[0], out_wh)


# --------------------------------------------------------------------------------------------------------------
# seqA tiling for the tiled hieroglyph detector
# seqA sequential content-adaptive tiling for the tiled hieroglyph detector, IN MEMORY (port of
# training/ppdoclayout/seqA_driver_2c.py, same rules, no files, no subprocess).
#
# Per line (short side already 256): tile [offset, offset+TILE] along the long axis, predict, keep score >= SKEEP,
# drop GIANT boxes (long extent > GIANT_RATIO * short unless score >= GIANT_SCORE). Partition the boxes into quadrat
# columns along the long axis. Commit every column before the first one touching the tile edge (the probably cropped
# quadrat), restart the next tile PAD px before it so that quadrat is re-seen whole. Fallbacks: no boxes -> step
# TILE-PAD, nothing touches the edge ("blank") -> restart after the last column, one column spans the tile ("e0") ->
# plain overlap step TILE-128. Cross-tile dedup (IoU > 0.5 or IoS > 0.6 vs committed), final IoU-0.5 NMS.
#     boxes = seqa(img_rgb, detector)   -> [(score, [x1,y1,x2,y2], label), ...] in the line-crop pixels
# --------------------------------------------------------------------------------------------------------------

PARAMS = dict(TILE=2048, PAD=5, MARGIN=6, SKEEP=0.30, GAPMIN=1, GIANT_RATIO=1.8, GIANT_SCORE=0.6)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def ios(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    m = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / m if m > 0 else 0


def _columns(boxes, la, gapmin):
    idx = sorted(range(len(boxes)), key=lambda k: boxes[k][1][la])
    cols = []
    for k in idx:
        b = boxes[k][1]; lo, hi = b[la], b[la + 2]
        if cols and lo - cols[-1][1] < gapmin:
            cols[-1][1] = max(cols[-1][1], hi); cols[-1][2].append(k)
        else:
            cols.append([lo, hi, [k]])
    return cols


def seqa(arr, detector, p=PARAMS, max_rounds=30):
    TILE, PAD, MARGIN, SKEEP = p["TILE"], p["PAD"], p["MARGIN"], p["SKEEP"]
    H, W = arr.shape[:2]; row = H <= W
    L = W if row else H; short = H if row else W; la = 0 if row else 1
    off, committed, rnd = 0, [], 0
    while True:
        crop = arr[:, off:min(off + TILE, L)] if row else arr[off:min(off + TILE, L), :]
        boxes = []
        for lab, sc, x1, y1, x2, y2 in detector(crop):
            if sc < SKEEP:
                continue
            b = [x1 + off, y1, x2 + off, y2] if row else [x1, y1 + off, x2, y2 + off]
            boxes.append((float(sc), [float(v) for v in b], int(lab) + 1))
        boxes = [(sc, b, cl) for sc, b, cl in boxes
                 if (b[la + 2] - b[la]) <= p["GIANT_RATIO"] * short or sc >= p["GIANT_SCORE"]]
        last = off + TILE >= L; edge = off + TILE
        status = [2] * len(boxes); done = False
        if last:
            done = True
        elif not boxes:
            off = off + TILE - PAD
        else:
            cols = _columns(boxes, la, p["GAPMIN"])
            e = next((ci for ci, c in enumerate(cols) if c[1] >= edge - MARGIN), None)
            if e is None:
                last_hi = max(c[1] for c in cols)
                new_off = int(last_hi) - PAD
                off = new_off if new_off > off else off + TILE - 128
            elif e == 0:
                for k, (sc, b, cl) in enumerate(boxes):
                    if b[la + 2] >= edge - MARGIN:
                        status[k] = 1
                off = off + TILE - 128
            else:
                for ci in range(e, len(cols)):
                    for k in cols[ci][2]:
                        status[k] = 1
                new_off = int(cols[e][0]) - PAD
                off = new_off if new_off > off else off + TILE - 128
        prev = [cb for _, cb, _ in committed]
        for k, (sc, b, cl) in enumerate(boxes):
            if status[k] == 2 and any(iou(b, cb) > 0.5 or ios(b, cb) > 0.6 for cb in prev):
                status[k] = 0
        committed += [boxes[k] for k in range(len(boxes)) if status[k] == 2]
        rnd += 1
        if done or rnd > max_rounds:
            break
    keep = []
    for sc, b, cl in sorted(committed, key=lambda x: -x[0]):
        if all(iou(b, k[1]) < 0.5 for k in keep):
            keep.append((sc, b, cl))
    return keep
