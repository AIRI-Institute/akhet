"""Human-in-the-loop text stage: OCR page -> editable text blocks -> transliteration -> translation.

A TextBlock is plain data. Every field can be edited by hand between stages, saved to JSON and loaded back:
    blocks = text_blocks(page)                  # one block per page text block, lines in reading order
    blocks[0].hieroglyphs = "G17 D21 D37 ..."   # fix the sign text by hand
    tr.transliterate(blocks)                    # fills .transliteration (only where empty, unless overwrite=True)
    blocks[0].transliteration = "..."           # fix the transliteration by hand
    tr.translate(blocks, "en")                  # fills .translation["en"]
    save_blocks(blocks, "blocks.json"); blocks = load_blocks("blocks.json")
A block can also be created from typed text alone: TextBlock(hieroglyphs="A1 G17 N35").

Sign text format = what the translation model was trained on: Gardiner codes separated by single spaces, TLA spelling
(Aa1 not AA1), a damaged sign as //, non-egypt placeholders dropped, lines of a block joined in reading order.
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .layout import NE_TOKEN


def tla_code(code):
    """classifier code -> TLA / translation-model spelling"""
    if code == "DAMAGED":
        return "//"
    if code.startswith("AA"):
        return "Aa" + code[2:]
    return code


@dataclass
class TextBlock:
    hieroglyphs: str = ""                                   # space-separated Gardiner (editable)
    transliteration: Optional[str] = None                   # editable
    translation: Dict[str, str] = field(default_factory=dict)   # {"en": ..., "de": ...} editable
    lines: List[int] = field(default_factory=list)          # source line ids on the page (empty for typed text)
    mdc: List[str] = field(default_factory=list)            # packed MdC of the source lines, for reference
    block: Optional[int] = None


def line_text(line):
    return " ".join(tla_code(t) for t in line.tokens if t != NE_TOKEN)


def text_blocks(page):
    out = []
    for k, lines in enumerate(page.blocks()):
        out.append(TextBlock(hieroglyphs=" ".join(t for t in (line_text(l) for l in lines) if t),
                             lines=[l.id for l in lines], mdc=[l.mdc for l in lines], block=k))
    return out


def save_blocks(blocks, path):
    json.dump([asdict(b) for b in blocks], open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def load_blocks(path):
    return [TextBlock(**d) for d in json.load(open(path, encoding="utf-8"))]
