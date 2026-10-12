---
license: cc-by-nc-sa-4.0
library_name: onnx
pipeline_tag: image-to-text
tags:
  - egyptology
  - hieroglyphs
  - ocr
  - manuel-de-codage
  - transliteration
  - translation
---

# akhet models

Models behind [akhet](https://github.com/AIRI-Institute/akhet), which reads Ancient Egyptian hieroglyphs from images: text lines, signs, Gardiner codes, reading order as Manuel de Codage, transliteration and translation, with a human in the loop at every step.

```bash
pip install "akhet[gpu,hub,translate] @ git+https://github.com/AIRI-Institute/akhet#subdirectory=pipeline"
```
```python
from akhet import Akhet, text_blocks, Translator
page = Akhet.from_pretrained()("page.jpg")          # OCR, editable
blocks = text_blocks(page)                         # editable text blocks
tr = Translator.from_pretrained()
tr.transliterate(blocks)
tr.translate(blocks, "en")
```

## Files

| file | role | input | size |
|---|---|---|---|
| `lines.onnx` | text line detector (RT-DETR, PP-DocLayout-plus-L) | page, keep-aspect resize + pad to 1024 | 125 MB |
| `glyphs_row.onnx`, `glyphs_col.onnx` | hieroglyph detector (RT-DETR), one per orientation | 256 x 2048 / 2048 x 256 line tile | 124 MB each |
| `direction_nonegypt.onnx` | line reading direction + modern text (RT-DETR) | line crop, resize + pad to 1024 | 125 MB |
| `sign_classifier.onnx` | sign embedding (DINOv2-base, ArcFace) | 224 x 224 sign crop | 330 MB |
| `sign_bank.npy`, `sign_bank.json` | one prototype per Gardiner sign (980 signs) | | 1 MB |
| `nllb/` | fine-tuned NLLB-1.3B, fp16: hieroglyphs to transliteration to English / German. Ships `tokenizer.json`: load it as is, transformers 5.x would otherwise rebuild the extended vocabulary with shifted ids | Gardiner text, TLA spelling | 2.6 GB |

The detectors output raw RT-DETR queries (boxes + class logits). The akhet package holds the exact pre- and post-processing.

## Held-out results

| model | test | result |
|---|---|---|
| lines | 5-fold cross-validation, 620 pages | F1@0.5 0.84 |
| lines | external benchmark, 84 pages | F1@0.5 0.82 |
| hieroglyphs | 141 real lines | AP@0.5 0.96, AP@0.75 0.86 |
| direction + modern text | 150 real lines | direction accuracy 0.91, modern-text AP@0.5 0.73 |
| sign classifier | real sign crops, 3 seeds | top-1 0.86, macro-F1 0.69 |
| sign classifier | signs absent from real training data | top-10 0.86 |
| translation model | TLA test clauses (400 / 265 / 300) | hieroglyphs to transliteration chrF 59.0 BLEU 60.9, to English via transliteration chrF 45.4 BLEU 25.1, to German chrF 51.9 BLEU 33.0 |

After evaluation the detectors and the classifier were retrained on all annotated data with the same recipe. These released weights are those final models. Converted to ONNX, they reproduce the original PaddleDetection and PyTorch models within 0.001 on every metric above. The fp16 translation model gives the same output as the fp32 original.

## Training data

Annotated page photographs and hand copies of hieroglyphic inscriptions, open-access museum images, synthetic hieroglyph lines rendered from the Thesaurus Linguae Aegyptiae with a learned layout model, and sign crops from public Egyptian hieroglyph datasets. Sequence and translation data come from the [Thesaurus Linguae Aegyptiae](https://thesaurus-linguae-aegyptiae.de) (CC BY-SA 4.0).

## License

CC BY-NC-SA 4.0: non-commercial use only, derived work under the same license. Part of the image training data is licensed for non-commercial use, and the translation model builds on NLLB-200 (CC BY-NC 4.0).

## Limitations

Trained on monumental hieroglyphic inscriptions, not hieratic. A sign outside the 980-sign inventory is mapped to the closest known sign. Transliteration and translation are drafts for a scholar to check.
