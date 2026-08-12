"""Inference script for the Akhet NLLB translation model.

Translates Middle Egyptian (hieroglyphs or transliteration) to English or German
using a fine-tuned NLLB/M2M-100 checkpoint. See README.md in this directory for
model details and the Hugging Face repo link.
"""
import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

LANG_TO_NLLB_TOKEN = {
    "hieroglyphs": "egy_Hiero",
    "transliteration": "egy_Tnt",
    "english": "eng_Latn",
    "german": "deu_Latn",
}


class NLLBTranslator:
    def __init__(self, model_path, max_length=4096, num_beams=4, repetition_penalty=3.0, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.max_length = max_length
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty

    def translate(self, text, src_lang, tgt_lang):
        self.tokenizer.src_lang = LANG_TO_NLLB_TOKEN[src_lang]
        tgt_token_id = self.tokenizer.convert_tokens_to_ids(LANG_TO_NLLB_TOKEN[tgt_lang])

        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=False,
                return_tensors="pt",
            ).to(self.device)

            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=tgt_token_id,
                num_beams=self.num_beams,
                early_stopping=True,
                max_length=self.max_length,
                repetition_penalty=self.repetition_penalty,
            )
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="kesha-humonen/egyptian-nllb",
                         help="Local path or Hugging Face repo id of the checkpoint")
    parser.add_argument("--text", required=True, help="Input text to translate")
    parser.add_argument("--src_lang", required=True, choices=sorted(LANG_TO_NLLB_TOKEN),
                         help="Source: hieroglyphs or transliteration")
    parser.add_argument("--tgt_lang", required=True, choices=sorted(LANG_TO_NLLB_TOKEN),
                         help="Target: english or german")
    args = parser.parse_args()

    translator = NLLBTranslator(args.model_path)
    print(translator.translate(args.text, args.src_lang, args.tgt_lang))


if __name__ == "__main__":
    main()
