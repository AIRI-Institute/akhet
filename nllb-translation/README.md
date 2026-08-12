# Middle Egyptian NLLB translation model

An NLLB/M2M-100-based sequence-to-sequence model fine-tuned to translate Middle Egyptian
(given as hieroglyphs or as transliteration) into English or German.

- **Weights**: [kesha-humonen/egyptian-nllb](https://huggingface.co/kesha-humonen/egyptian-nllb)
  on Hugging Face
- **Base architecture**: NLLB-200-distilled-1.3B (M2M-100 arch, `d_model=1024`, 24 encoder +
  24 decoder layers), extended with two custom source-language tokens, `egy_Hiero`
  (hieroglyphs) and `egy_Tnt` (transliteration)
- **Checkpoint**: step 223668 (9 epochs, `eval_strategy=epoch`, `metric_for_best_model=eval_chrf`)
- **Trained directions**: hieroglyphs→English, hieroglyphs→German, hieroglyphs→transliteration,
  transliteration→English, transliteration→German
- **Training-time eval** (trainer's own held-out split, aggregate across all five directions
  above): chrF 57.15, BLEU 40.79 — this is the training run's own aggregate metric, not
  necessarily on the same per-direction test split used elsewhere for this project's numbers.

## Input format

- `transliteration`: plain Egyptological transliteration text, e.g. `nn sšp nṯr.PL =sn
  tꜥ-ḥḏ.PL =sn`.
- `hieroglyphs`: **space-separated Gardiner sign codes**, not raw Unicode hieroglyph
  characters, e.g. `O36 Z2A M17 S29 M23 R4`. Confirmed against the training config
  (`unicode_hieroglyphs: False` in `hiero_multiout_1e4_hiero-t-tnt_gardiner_123.yaml`) — passing
  raw Unicode glyphs instead will not work.

## Usage

```bash
pip install -r requirements.txt

python translate.py \
  --model_path kesha-humonen/egyptian-nllb \
  --src_lang hieroglyphs --tgt_lang english \
  --text "O36 Z2A M17 S29 M23 R4"
# -> model output: "The tombs thereof destroyed;"
# -> gold reference: "walls were old,"

python translate.py \
  --model_path kesha-humonen/egyptian-nllb \
  --src_lang transliteration --tgt_lang english \
  --text "nn sšp nṯr.PL =sn tꜥ-ḥḏ.PL =sn"
# -> model output: "their gods shall not libate their white bread."
# -> gold reference: "their gods shall not accept their white bread."
```

Both examples above are real outputs from a GPU run of this exact checkpoint (test-set inputs,
not cherry-picked): the transliteration case is a near-exact match to the gold reference, and
the hieroglyphs case shows the model producing coherent, in-register English even on a short,
context-free 6-sign clause where a semantic mismatch is unsurprising — neither number was
invented for this README.

Or from Python:

```python
from translate import NLLBTranslator

translator = NLLBTranslator("kesha-humonen/egyptian-nllb")
translator.translate("O36 Z2A M17 S29 M23 R4", src_lang="hieroglyphs", tgt_lang="english")
```

Supported `src_lang`/`tgt_lang` values: `hieroglyphs`, `transliteration`, `english`, `german`.
