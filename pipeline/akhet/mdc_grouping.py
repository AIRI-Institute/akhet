"""MdC quadrat grouping + reading order for ONE line of hieroglyph boxes (akhet).
Logic vendored UNCHANGED from egyvlm scripts/mine_grouping.py (the miner that produced our annotated-line MdC):
column-first quadrat segmentation with look-ahead, recursive */: split by box gaps, insertions (HOST&SIGN,
HOST&&&(group)), cartouche <1-..-0> / serekh <S-..-> by code. Only the dataset loader and the viz were left out.
Not thread-safe: the reading direction lives in the module global _XDIR (set per line by mine_line).
"""
import re
import numpy as np
# code normalization reused from make_glyph_pairs.py (norm + variants): strip *var*/*damaged*/*hierat*/*?*
# flags, rotation \r1, trailing \ / v V; fix AA->Aa casing; keep only valid Gardiner codes, else drop the sign.
_GARD = re.compile(r"^(A|B|C|D|E|F|G|H|I|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z|Aa|NL|NU)\d+[A-Za-z]?$")
def norm_code(t):
    if not t:
        return None
    t = re.sub(r"\*[^*]*\*?", "", str(t).strip())   # *damaged* *var* *hierat* *?* flags
    t = re.sub(r"\\r[0-9]*", "", t)                  # rotation code \r1
    t = re.sub(r"[\\/vV]+$", "", t).strip()          # trailing \ / v V (rotation/variant markers)
    if t[:2] == "AA":
        t = "Aa" + t[2:]                             # AA1 -> Aa1
    return t if _GARD.match(t) else None

UNDER_W = 0.35                                                 # squareness: weight on UNDERFILL vs OVERFILL. A cell
                                                              # a bit under 1 square is normal (cheap); a cell stretched
                                                              # PAST 1 square is a monster (full penalty).
# FRAG = a unit whose READING-AXIS extent is < this fraction of the cell is a fragment that must lean on a neighbour.
# The reading axis differs by orientation, so ONE constant means two things (Maksim: "for * care about thin, for :
# care about short, don't confuse them"): ROW reading-axis = WIDTH/height -> a THIN sign is the fragment (this caps
# too-long `*` runs); COLUMN reading-axis = HEIGHT/width -> a SHORT sign is the fragment (caps too-long `:` runs).
FRAG = 0.45
FRAG_PEN = 3.0                                                 # S29 width=0.36cell merges (thin); D21:D28*Z1 height=0.54cell stands.
MAX_EXT = 1.60   # quadrat long-extent cap (x short). COLUMN-first packing enforces a real width cap (old sign-greedy bypassed it via overlap-exc). 1.60 sits between narrow+stack (MEH132 S29*(X1:N35)=1.51sq KEEP) and cell+cell (MEH69 D60|stack=1.79sq SPLIT).
MARKERS = set("/\\?<>.")
# TLA-derived (tla_multi_language_parsed.json): signs that host a &&& / ^^^ inserted group
INSERTION = {"F20", "I10", "G5", "G1", "F40", "G39", "E17", "F20A", "I1", "V22", "I3A", "V12", "F30",
             "G17", "W54", "E9", "G43", "Z3", "U36", "Z2A", "X2A", "S34", "W10", "U22", "Z7", "N29", "Z2B", "O29v"}
ENCLOSURE = {"V10", "V11", "O33"}     # cartouche (V10/V11), serekh (O33)  -- image-side enclosure signs
CONTAIN_MIN = 0.6                      # fraction of a sign's area that must be inside a container to SEED an insertion
CONTAIN_SIB = 0.35                     # once a host is a CONFIRMED insertion, weaker siblings (>= this) join its pocket
INS_SIZE = 0.70                        # an inserted sign must be < this fraction of the host's area (small-in-big)
_XDIR = 1                              # reading direction on the x axis: +1 left->right, -1 right->left ("Left" rows)


def area(s): return max(1.0, (s["x2"] - s["x1"]) * (s["y2"] - s["y1"]))
def frac_inside(big, sm):
    ix1, iy1 = max(big["x1"], sm["x1"]), max(big["y1"], sm["y1"])
    ix2, iy2 = min(big["x2"], sm["x2"]), min(big["y2"], sm["y2"])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1) / area(sm)


def _balanced(m):
    d = 0
    for c in m:
        d += (c == "(") - (c == ")")
        if d < 0: return False
    return d == 0
def strip_outer(m):
    return m[1:-1] if len(m) > 1 and m[0] == "(" and m[-1] == ")" and _balanced(m[1:-1]) else m


PREC = {"*": 2, ":": 1}                            # MdC precedence: * binds TIGHTER than : (A:B*C == A:(B*C))
def arrange(units):
    """Recursive */: split by dominant clean gap. Returns (mdc, top_op, leaves, min_gap). MINIMAL parens:
    a child is wrapped only when its operator binds LOOSER than the parent's (e.g. a `:` group under `*`)."""
    if len(units) == 1:
        u = units[0]; return u["mdc"], None, list(u["leaves"]), 1e9   # leaf/atomic -> op None, never wrapped
    cand = []
    for axis, a1, a2 in [("y", "y1", "y2"), ("x", "x1", "x2")]:
        s = sorted(units, key=lambda z: (z[a1] + z[a2]) / 2.0)
        for i in range(1, len(s)):
            # TRUE guillotine gap: the cut is clean only if EVERY left sign ends before EVERY right sign
            # starts. Using s[i][a1] as the right edge is wrong -- a wide sign that sorts later by center can
            # span back across the cut (Aa15/Y5 over a W24 row), faking a positive gap and a bogus `*` split.
            egap = min(z[a1] for z in s[i:]) - max(z[a2] for z in s[:i])
            cgap = (s[i][a1] + s[i][a2] - s[i - 1][a1] - s[i - 1][a2]) / 2.0   # center separation at this cut
            cand.append((egap, cgap, axis, i, s))
    # Prefer a CLEAN guillotine cut (widest positive edge gap). If EVERY cut overlaps (fully interlocking signs,
    # e.g. MEH69 U1 shifted into Aa11/D36), edge gaps are all noise -> pick the axis of largest CENTER separation,
    # which recovers the true column split (Aa11:D36 | U1) instead of interleaving U1 into a `:` stack.
    clean = [c for c in cand if c[0] >= 0]
    gap, cgap, axis, i, s = max(clean, key=lambda c: c[0]) if clean else max(cand, key=lambda c: c[1])
    op = ":" if axis == "y" else "*"
    ls, lop, ll, lg = arrange(s[:i]); rs, rop, rl, rg = arrange(s[i:])
    if axis == "x" and _XDIR < 0:                  # right-to-left row: the right group is read/written first
        ls, lop, ll, rs, rop, rl = rs, rop, rl, ls, lop, ll
    def w(cs, co): return f"({cs})" if (co is not None and PREC[co] < PREC[op]) else cs
    return f"{w(ls, lop)}{op}{w(rs, rop)}", op, ll + rl, min(gap, lg, rg)


def build_units(signs):
    """Resolve containers into single units (recursively). INSERTION detected GEOMETRICALLY (a bigger sign
    holds a smaller sign well inside its bbox); ENCLOSURE (cartouche/serekh) gated by CODE. HOST-FIRST order:
    HOST^^^grp (top) / HOST&sign (single below) / HOST&&&(grp) (group below)."""
    for s in signs:
        s.setdefault("mdc", s["t"]); s.setdefault("leaves", [s["t"]]); s.setdefault("members", [s]); s.setdefault("is_container", False)
    consumed = set()
    for c in sorted(signs, key=lambda s: -area(s)):       # biggest-first; ANY sign can host / enclose
        if id(c) in consumed:
            continue
        avail = [s for s in signs if s is not c and id(s) not in consumed and area(s) < INS_SIZE * area(c)]
        seed = [s for s in avail if frac_inside(c, s) >= CONTAIN_MIN]
        if not seed:
            continue
        # Two-tier: 0.6 SEEDS the host. A confirmed insertion (not a code-gated enclosure) then also grabs weaker
        # SIBLINGS (>= CONTAIN_SIB) sharing the pocket -- a horizontal pair under a snake tail hangs at different
        # heights (MEH15 D56=0.65 seeds, D54=0.41 joins), and they read as ONE inserted group.
        inside = seed if c["t"] in ENCLOSURE else [s for s in avail if frac_inside(c, s) >= CONTAIN_SIB]
        inner = build_units(inside)                       # recurse -> handles nested containers
        gm, _, gl, _ = arrange(inner)
        if c["t"] in ENCLOSURE:                           # cartouche / serekh by CODE
            c["mdc"] = f"<1-{gm}-0>" if c["t"] in ("V10", "V11") else f"<S-{gm}->"
            c["leaves"] = [c["t"]] + gl
        elif len(inside) == 1:                            # SINGLE inserted sign -> ligature &
            c["mdc"] = f"{c['t']}&{gm}"; c["leaves"] = [c["t"]] + gl
        else:                                             # GROUP inserted -> HOST&&&(group)  (TLA: &&&( 963x; group-^^^ ~0)
            c["mdc"] = f"{c['t']}&&&({gm})"; c["leaves"] = [c["t"]] + gl
        c["is_container"] = True
        c["members"] = [c] + [m for s in inside for m in s["members"]]
        for s in inside:
            consumed.add(id(s))
    return [s for s in signs if id(s) not in consumed]


def seg_quadrats(units, is_row, short, gap_break=0.45, col_frac=0.5):
    # COLUMN-FIRST segmentation. A sign-by-sign greedy width cap over-merges: a stack's first sign can
    # squeak under the cap and pull the whole stack into a neighbour's quadrat (MEH69 D60*(stack)). Instead:
    # (1) group signs into COLUMNS by reading-axis overlap (a ':' stack is one column), then
    # (2) pack whole COLUMNS into ~square quadrats -- so a column that already fills a square is never
    #     glued to an adjacent sign; they split at the clean column boundary (-> D60 - stack).
    lo = (lambda s: s["x1"]) if is_row else (lambda s: s["y1"])
    hi = (lambda s: s["x2"]) if is_row else (lambda s: s["y2"])
    units = sorted(units, key=lambda s: (lo(s) + hi(s)) / 2.0)
    # (1) build columns: a sign joins the current column if it overlaps it by >= col_frac of the narrower extent
    columns = []
    for s in units:
        joined = False
        if columns:
            c = columns[-1]; c_lo = min(lo(u) for u in c); c_hi = max(hi(u) for u in c)
            ov = min(hi(s), c_hi) - max(lo(s), c_lo)
            wmin = min(hi(s) - lo(s), c_hi - c_lo)
            if ov >= col_frac * max(wmin, 1.0):
                c.append(s); joined = True
        if not joined:
            columns.append([s])
    # (2) pack columns into quadrats. Each boundary is decided by SQUARENESS, not raw gap: a bridging column
    #     joins whichever side yields the more-square PAIR of quadrats. sqerr is ASYMMETRIC -- OVERFILL (a cell
    #     stretched past 1 square = a cramped "monster") is penalised in full, UNDERFILL is cheap (a lone small
    #     sign is a perfectly normal quadrat, 73% of TLA). Without that asymmetry the packer fears small cells
    #     and stacks a standalone full cell (the I10 snake, 0.76sq) onto its neighbour -> 1.30sq monster.
    def sqerr(span):
        r = span / short
        if r > 1.0:
            return r - 1.0                                    # overfill: monster, full penalty
        e = UNDER_W * (1.0 - r)                               # underfill: cheap (a small cell is fine)
        if r < FRAG:
            e += FRAG_PEN * (FRAG - r)                        # ...but a THIN fragment alone is bad -> wants a partner
        return e
    cols, cur, qstart, cover, min_gap = [], list(columns[0]), min(lo(u) for u in columns[0]), max(hi(u) for u in columns[0]), 1e9
    for i in range(1, len(columns)):
        col = columns[i]
        c_lo = min(lo(u) for u in col); c_hi = max(hi(u) for u in col)
        gap = c_lo - cover; would = c_hi - qstart
        if gap > gap_break * short or would > MAX_EXT * short:
            brk = True                                       # hard gap boundary or square-cap overflow
        else:
            pair = None                                      # if the NEXT column can pair with col, weigh that
            if i + 1 < len(columns):
                nxt = columns[i + 1]; n_lo = min(lo(u) for u in nxt); n_hi = max(hi(u) for u in nxt)
                if (n_hi - c_lo) <= MAX_EXT * short: pair = (n_lo, n_hi)
            if pair:
                n_lo, n_hi = pair
                err_merge = sqerr(c_hi - qstart) + sqerr(n_hi - n_lo)     # cur+col , next alone
                err_break = sqerr(cover - qstart) + sqerr(n_hi - c_lo)    # cur alone, col+next
            else:                                            # last column (or next can't pair): col stands alone
                err_merge = sqerr(c_hi - qstart)                         # cur+col as one quadrat
                err_break = sqerr(cover - qstart) + sqerr(c_hi - c_lo)   # cur alone + col alone
            brk = err_break < err_merge
        if brk:
            min_gap = min(min_gap, gap); cols.append(cur); cur = list(col); qstart = c_lo; cover = c_hi
        else:
            cur.extend(col); cover = max(cover, c_hi)
    cols.append(cur)
    return cols, (min_gap if len(cols) > 1 else short)


def mine_line(ln):
    signs = ln["signs"]                                # already norm_code-cleaned in load_lines
    is_row = "Row" in ln["label"] or (ln["lbbox"] and (ln["lbbox"]["x2"] - ln["lbbox"]["x1"]) >= (ln["lbbox"]["y2"] - ln["lbbox"]["y1"]))
    b = ln["lbbox"]
    short = ((b["y2"] - b["y1"]) if is_row else (b["x2"] - b["x1"])) if b else float(np.median([(s["y2"] - s["y1"]) if is_row else (s["x2"] - s["x1"]) for s in signs]))
    short = max(1.0, short)
    global _XDIR
    # "Right" = right-to-left reading. The horizontal `*` operand order is R->L for BOTH rows AND columns
    # (a column read R->L still has its side-by-side signs written rightmost-first), so _XDIR keys on dir alone.
    _XDIR = -1 if ln.get("dir") == "Right" else 1
    units = build_units(signs)
    cols, seg_gap = seg_quadrats(units, is_row, short)
    if is_row and _XDIR < 0:
        cols = cols[::-1]                                          # ROW quadrats run along x -> reverse for R->L;
                                                                  # COLUMN quadrats run top->bottom -> never reversed
    quad_mdc, quad_leaves, quad_sizes, conf_min = [], [], [], 1e9
    for c in cols:
        m, _, leaves, g = arrange(c); quad_mdc.append(m); quad_leaves += leaves; quad_sizes.append(len(leaves)); conf_min = min(conf_min, g)
    packed = "-".join(quad_mdc); plain = "-".join(quad_leaves)
    n_cont = sum(1 for u in units if u.get("is_container"))
    biggest = max(len(c) for c in cols)
    biggest_leaves = max(quad_sizes)                              # SIGNS in the fullest quadrat (leaves, not units)
    # Confidence = likelihood the PACKING is faithful (box positions are always exact). Re-tuned 2026-08-26: the old
    # any-negative-gap penalty is stale -- interlocking is now RESOLVED by arrange's center fallback, so a negative
    # edge-gap is normal in dense real script. Penalise only GENUINE ambiguity, scale-relative to the cell (short).
    conf = 1.0
    if biggest_leaves > 6: conf -= 0.3                            # dense (<=6 flats is plausible) ...
    if biggest_leaves > 10: conf -= 0.4                           # ... but a very full quadrat is a likely over-merge
    if conf_min < -0.5 * short: conf -= 0.3                       # SEVERE interlocking: the fallback had to guess
    if seg_gap < -0.35 * short: conf -= 0.2                       # neighbouring quadrats heavily overlap in reading axis
    conf = max(0.0, conf)
    return {"src": ln["src"], "img": ln["img"], "lidx": ln["lidx"], "is_row": is_row, "dir": ln["dir"],
            "plain": plain, "packed": packed, "n_signs": len(signs), "n_quads": len(cols),
            "n_containers": n_cont, "biggest_quad": biggest, "conf": round(conf, 2),
            "quad_sizes": quad_sizes, "cols": cols, "lbbox": ln["lbbox"]}
