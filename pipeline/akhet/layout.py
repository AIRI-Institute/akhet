"""Layout rules on top of the detectors: non-egypt vs hieroglyph conflict, reading order inside a line (MdC), and the
reading order of lines on a page.

conflict      NON-EGYPT WINS: non-egypt boxes are kept whole, a hieroglyph is removed when the share of ITS OWN area
              inside a non-egypt box is >= thr (0.5). Small touches never remove a hieroglyph.
order_line    quadrat grouping + reading order + packed MdC (mdc_grouping = our annotated-line miner). "Right" lines read
              right to left, "Left" / "Unknown" / None left to right, columns top to bottom. NE_TOKEN marks where a
              non-egypt box sits in the reading order.
order_lines   page line order: recursive XY-cut into blocks (a gap is a block boundary only if >= BLOCK_GAP x the median
              line thickness), rows top-down, columns along the majority direction (Right -> right to left).
"""
import statistics
from . import mdc_grouping as M

NE_TOKEN = "{NE}"
BLOCK_GAP = 1.0


# ----------------------------------------------------------------------------------------------------- conflict
def overlap_share(g, n):
    iw = min(g[2], n[2]) - max(g[0], n[0]); ih = min(g[3], n[3]) - max(g[1], n[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / ((g[2] - g[0]) * (g[3] - g[1]))


def remove_signs_inside_non_egypt(sign_boxes, ne_boxes, thr=0.5):
    """-> (kept indices, removed indices)"""
    keep, removed = [], []
    for i, g in enumerate(sign_boxes):
        share = max((overlap_share(g, n) for n in ne_boxes), default=0.0)
        (removed if share >= thr else keep).append(i)
    return keep, removed


# ----------------------------------------------------------------------------------------- reading order in a line
def _center(b, axis):
    return (b[axis + "1"] + b[axis + "2"]) / 2.0


def _split_packed(packed, n):
    """split packed MdC into its n top-level quadrats ('-' also appears inside <1-..-0> / <S-..-> enclosures)."""
    parts, cur, depth = [], "", 0
    for ch in packed:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "-" and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
    parts.append(cur)
    assert len(parts) == n, f"packed split {len(parts)} != {n} quadrats: {packed}"
    return parts


def order_line(signs, is_row, direction="Left", line_box=None, non_egypt=()):
    """signs: [{"code", "x1", "y1", "x2", "y2"}] -> dict(order, codes, tokens, packed, quads, n_quads, conf, direction_used)"""
    xdir = -1 if direction == "Right" else 1
    ne = [dict(zip(("x1", "y1", "x2", "y2"), map(float, b))) for b in non_egypt]
    used = "Right" if xdir < 0 else "Left"
    if not signs:
        return {"order": [], "codes": [], "tokens": [NE_TOKEN] * len(ne), "packed": "-".join([NE_TOKEN] * len(ne)),
                "quads": [], "n_quads": 0, "conf": 0.0, "direction_used": used}
    work = [{"t": s["code"], "x1": float(s["x1"]), "y1": float(s["y1"]), "x2": float(s["x2"]), "y2": float(s["y2"]), "_i": i}
            for i, s in enumerate(signs)]
    lb = dict(zip(("x1", "y1", "x2", "y2"), map(float, line_box))) if line_box is not None else None
    r = M.mine_line({"src": "", "img": "", "lidx": 0, "label": "Row" if is_row else "Column", "dir": used,
                     "lbbox": lb, "signs": work})
    leaves = r["plain"].split("-")
    quads, cut = [], 0
    for units, n in zip(r["cols"], r["quad_sizes"]):
        members, seen = [], set()          # nested containers list inner signs twice in `members` (leaves are right)
        for u in units:
            for m in u["members"]:
                if id(m) not in seen:
                    seen.add(id(m)); members.append(m)
        q = []
        for code in leaves[cut:cut + n]:
            cand = sorted((m for m in members if m["t"] == code), key=lambda m: (_center(m, "y"), xdir * _center(m, "x")))
            m = cand[0]; members.remove(m); q.append(m["_i"])
        assert not members, "leaf/member mismatch inside a quadrat"
        quads.append(q); cut += n
    order = [i for q in quads for i in q]
    assert sorted(order) == list(range(len(signs)))
    rpos = lambda b: xdir * _center(b, "x") if is_row else _center(b, "y")
    qpos = [sum(rpos(work[i]) for i in q) / len(q) for q in quads]
    slots = [[] for _ in range(len(quads) + 1)]
    for b in sorted(ne, key=rpos):
        slots[next((j for j, p in enumerate(qpos) if p > rpos(b)), len(quads))].append(NE_TOKEN)
    pieces = _split_packed(r["packed"], len(quads))
    tokens, parts = [], []
    for j, q in enumerate(quads):
        tokens += slots[j] + [signs[i]["code"] for i in q]; parts += slots[j] + [pieces[j]]
    tokens += slots[-1]; parts += slots[-1]
    return {"order": order, "codes": [signs[i]["code"] for i in order], "tokens": tokens, "packed": "-".join(parts),
            "quads": quads, "n_quads": len(quads), "conf": r["conf"], "direction_used": used}


# ------------------------------------------------------------------------------------------- line order on a page
def _majority(dirs):
    return "Right" if dirs.count("Right") > dirs.count("Left") else "Left"


def _thick(ln):
    b = ln["box"]
    return (b[3] - b[1]) if ln["is_row"] else (b[2] - b[0])


def _gaps(idx, lines, ax):
    iv = sorted((lines[i]["box"][ax], lines[i]["box"][ax + 2]) for i in idx)
    out, end = [], iv[0][1]
    for lo, hi in iv[1:]:
        if lo > end:
            out.append((lo - end, (lo + end) / 2.0))
        end = max(end, hi)
    return out


def _order_block(idx, lines, page_dir):
    rows = [i for i in idx if lines[i]["is_row"]]; cols = [i for i in idx if not lines[i]["is_row"]]
    cy = lambda i: (lines[i]["box"][1] + lines[i]["box"][3]) / 2.0
    cx = lambda i: (lines[i]["box"][0] + lines[i]["box"][2]) / 2.0
    sgn_c = -1 if (_majority([lines[i]["direction"] for i in cols]) if cols else page_dir) == "Right" else 1
    sgn_r = -1 if (_majority([lines[i]["direction"] for i in rows]) if rows else page_dir) == "Right" else 1
    if len(rows) >= len(cols):
        return sorted(idx, key=lambda i: (cy(i), sgn_r * cx(i)))
    return sorted(idx, key=lambda i: (sgn_c * cx(i), cy(i)))


def order_lines(lines):
    """lines: [{"box", "is_row", "direction"}] -> (order, block)"""
    if not lines:
        return [], []
    page_dir = _majority([ln["direction"] for ln in lines])
    min_gap = BLOCK_GAP * statistics.median(_thick(ln) for ln in lines)
    blocks = []

    def cut(idx):
        best = None
        for ax in (1, 0):
            for g, pos in _gaps(idx, lines, ax):
                if g >= min_gap and (best is None or g > best[0]):
                    best = (g, pos, ax)
        if best is None:
            blocks.append(_order_block(idx, lines, page_dir)); return
        _, pos, ax = best
        a = [i for i in idx if (lines[i]["box"][ax] + lines[i]["box"][ax + 2]) / 2.0 < pos]
        b = [i for i in idx if (lines[i]["box"][ax] + lines[i]["box"][ax + 2]) / 2.0 >= pos]
        first, second = (a, b) if (ax == 1 or page_dir != "Right") else (b, a)
        cut(first); cut(second)

    cut(list(range(len(lines))))
    order, block = [], [0] * len(lines)
    for k, bl in enumerate(blocks):
        for i in bl:
            order.append(i); block[i] = k
    return order, block
