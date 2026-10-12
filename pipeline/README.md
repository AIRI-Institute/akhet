# akhet reading pipeline

**Read Ancient Egyptian hieroglyphs from a photo.** akhet finds the lines of text on a page, detects every sign, names it with its Gardiner code, reconstructs the reading order as Manuel de Codage, and hands the text to a translation model that produces a transliteration and an English or German translation.

Every step returns plain, editable data. An Egyptologist can correct the signs, the direction or the reading order by hand and continue from there, correct the transliteration and translate again. The machine does the tedious part and the expert keeps the last word.

```
page image
  └─ 1  lines              where the text lines are (rows and columns)
  └─ 2  signs              every hieroglyph in each line
  └─ 3  direction          left-to-right, right-to-left or unknown, plus modern (non-Egyptian) text
  └─ 4  sign names         Gardiner code for each sign, with ranked alternatives
  └─ 5  reading order      quadrats, packed Manuel de Codage, line order on the page, text blocks
  └─ 6  transliteration    per text block            ✎ editable
  └─ 7  translation        English or German         ✎ editable
       ✎ every stage can be corrected by hand before the next one runs
```

## Install

```bash
git clone https://github.com/AIRI-Institute/akhet
pip install "./akhet/pipeline[cpu,hub]"               # OCR on CPU
pip install "./akhet/pipeline[gpu,hub]"               # OCR on an NVIDIA GPU (CUDA 12 runtime included)
pip install "./akhet/pipeline[gpu,hub,translate]"     # + transliteration and translation (PyTorch)
```

Or straight from GitHub: `pip install "akhet[gpu,hub,translate] @ git+https://github.com/AIRI-Institute/akhet#subdirectory=pipeline"`.

The OCR runs on [ONNX Runtime](https://onnxruntime.ai) with numpy, Pillow and OpenCV, no deep learning framework needed. Translation uses PyTorch and Hugging Face Transformers.

## Quickstart

```python
from akhet import Akhet

ocr = Akhet.from_pretrained()              # downloads the models once (or Akhet("path/to/models"))
page = ocr("examples/sample_page.png")

for line in page.lines:
    print(line.page_index, line.direction, line.mdc)

page.draw("examples/sample_page.png").save("result.jpg")
page.to_json("result.json")
```

Each line carries its signs in reading order, and each sign keeps its box, its Gardiner code and the ranked alternatives:

```python
sign = page.lines[0].signs[0]
sign.code, sign.similarity      # ('G17', 0.555)
sign.candidates                 # [('G17', 0.555), ('Z5', 0.514), ('D20', 0.489), ...]
```

## Human in the loop

**Correct the OCR.** Save the page, fix it in any editor or annotation tool, load it back. Or fix it in Python. Then let akhet recompute the reading order, the Manuel de Codage and the text blocks from your corrections.

```python
from akhet import Page

page = Page.from_json("result.json")
page.lines[2].signs[4].code = "N35"        # wrong sign name
page.lines[5].direction = "Right"          # wrong reading direction
del page.lines[7].signs[0]                 # a false detection
page.relayout()                            # reading order, MdC, page order, blocks
```

**Transliterate, correct, translate.** The text stage works on text blocks. Each block holds plain strings you can edit.

```python
from akhet import text_blocks, save_blocks, load_blocks, Translator

blocks = text_blocks(page)                 # one block per text block on the page
blocks[0].hieroglyphs                      # 'G17 D21 D37 X1 ...'  (Gardiner, TLA spelling)

tr = Translator.from_pretrained()
tr.transliterate(blocks)                   # fills block.transliteration
blocks[0].transliteration = "..."          # your correction
tr.translate(blocks, "en")                 # fills block.translation["en"]  ("de" for German)
save_blocks(blocks, "blocks.json")
```

A stage only fills fields that are still empty, so running it again never overwrites your corrections (pass `overwrite=True` to redo them). Text typed by hand works too, no image needed:

```python
from akhet import TextBlock
blocks = [TextBlock(hieroglyphs="V30 N16 N16")]
tr.translate(tr.transliterate(blocks), "en")
blocks[0].translation["en"]                # 'lord of the two lands'
```

## Output

| field | meaning |
|---|---|
| `line.box`, `line.is_row` | line box on the page, row or column |
| `line.direction` | predicted reading direction: `Left` (read left to right), `Right` (right to left), `Unknown` |
| `line.signs` | signs in reading order: `box`, `score`, `code`, `similarity`, `candidates` |
| `line.mdc` | packed Manuel de Codage (`-` quadrat, `:` stacked, `*` side by side, `&` / `&&&` insertion, `<1-...-0>` cartouche) |
| `line.tokens` | Gardiner codes in reading order, `{NE}` where modern text interrupts the line |
| `line.non_egypt` | boxes of modern text (labels, numbers, captions) |
| `line.page_index`, `line.block` | position in the page reading order and the text block it belongs to |

`Unknown` lines are read left to right. Pass `ocr(image, direction="Right")` to force a direction for a whole page.

## Models

| model | role | architecture | held-out result |
|---|---|---|---|
| `lines.onnx` | text lines | RT-DETR (PP-DocLayout-plus-L), 1024 square | F1@0.5 0.84 (5-fold CV), 0.82 on an external benchmark of 84 pages |
| `glyphs_row.onnx`, `glyphs_col.onnx` | hieroglyph boxes | RT-DETR, sequential tiling of the line | AP@0.5 0.96 on 141 real lines |
| `direction_nonegypt.onnx` | line direction + modern text | RT-DETR, 1024 square | direction accuracy 0.91, modern-text AP@0.5 0.73 on 150 real lines |
| `sign_classifier.onnx` + `sign_bank` | Gardiner code | DINOv2-base metric embedding, nearest prototype over 980 signs | top-1 0.86, macro-F1 0.69 on real crops, 0.86 top-10 on signs never seen in real training data |
| `nllb/` | transliteration, translation | fine-tuned NLLB-1.3B on Thesaurus Linguae Aegyptiae clauses, the model of [`nllb-translation/`](../nllb-translation/) | TLA test clauses: transliteration chrF 59, English chrF 45 (via transliteration), German chrF 52 |

The released detectors and classifier were retrained on all annotated data after evaluation, with the same recipe. The ONNX models reproduce the original PaddleDetection and PyTorch models: every benchmark metric agrees within 0.001.

**Speed.** On an NVIDIA H100 a page of 8 lines and 300 signs takes about 7 seconds for the OCR. On CPU expect 5 to 90 seconds per page, depending on how many signs it holds.

## Limitations

- The models were trained mostly on hand copies and photographs of monumental hieroglyphic inscriptions. Hieratic and cursive hands are out of scope.
- The sign inventory has 980 Gardiner signs. A sign outside it gets the closest known sign, so check low similarity values.
- Line order on the page is geometric (blocks by whitespace, rows top to bottom, columns along the reading direction). Complex layouts mixing rows and columns may need a hand correction.
- Transliteration and translation are machine output for a scholar to check, not an edition.

## License

Code and model weights: [CC BY-NC-SA 4.0](LICENSE). Non-commercial use only. Derived work must keep the same license.

The text corpus behind the translation model and the sign sequences is the [Thesaurus Linguae Aegyptiae](https://thesaurus-linguae-aegyptiae.de) (CC BY-SA 4.0). The translation model builds on [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) (CC BY-NC 4.0). The sample page in `examples/` is public domain.

## Acknowledgements

Thesaurus Linguae Aegyptiae and the Berlin-Brandenburg Academy of Sciences and Humanities, PaddleDetection and PP-DocLayout, DINOv2, NLLB-200, ONNX Runtime.

## Citation

```bibtex
@inproceedings{humonen2025automatic,
  title={Automatic Image Translation of Long Ancient Egyptian Texts for Augmented Reality Applications},
  author={Humonen, Innokentiy and Golyadkin, Maksim and Kalin, Danil and Makarov, Ilya},
  booktitle={2025 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct)},
  pages={875--876},
  year={2025},
  organization={IEEE},
  doi={10.1109/ISMAR-Adjunct68609.2025.00234}
}

@inproceedings{humonen2026human,
  title={Human-in-the-Loop Egyptology: A System for Ancient Egyptian Text Study},
  author={Humonen, Innokentiy and Golyadkin, Maksim and Rubanova, Valeria and Alexandrova, Ekaterina and Loktionov, Alexandre A and Makarov, Ilya},
  booktitle={Proceedings of the Extended Abstracts of the 2026 CHI Conference on Human Factors in Computing Systems},
  pages={1--5},
  year={2026}
}
```
