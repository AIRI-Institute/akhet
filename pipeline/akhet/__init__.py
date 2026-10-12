"""akhet: reading Ancient Egyptian hieroglyphs from images.

    from akhet import Akhet
    ocr = Akhet.from_pretrained()            # or Akhet("path/to/models")
    page = ocr("page.jpg")
    for line in page.lines:
        print(line.direction, line.mdc)
"""
from .pipeline import Akhet
from .result import Page, Line, Sign
from .layout import NE_TOKEN
from .text import TextBlock, text_blocks, save_blocks, load_blocks

__version__ = "0.1.0"
__all__ = ["Akhet", "Page", "Line", "Sign", "NE_TOKEN", "TextBlock", "text_blocks", "save_blocks", "load_blocks", "Translator"]


def __getattr__(name):            # Translator needs the optional [translate] extra -> import it lazily
    if name == "Translator":
        from .translate import Translator
        return Translator
    raise AttributeError(name)
