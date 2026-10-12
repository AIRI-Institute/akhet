"""Transliteration + translation with the fine-tuned NLLB-1.3B (one model, all directions).
Needs the optional extra:  pip install akhet[translate]   (torch, transformers, sentencepiece)

    tr = Translator("models/nllb")
    tr.transliterate(blocks)          # hieroglyphs (egy_Hiero) -> transliteration (egy_Tnt)
    tr.translate(blocks, "en")        # transliteration (egy_Tnt) -> English (eng_Latn) / "de" German (deu_Latn)
    tr.run("A1 G17 N35", "hieroglyphs", "transliteration")   # raw call on one string

Decoding = the deployed old pipeline: beam search 4, repetition penalty 3.0, early stopping, max length 4096,
one text at a time. Stages only fill EMPTY fields unless overwrite=True, so hand edits are never lost.
"""
LANG = {"hieroglyphs": "egy_Hiero", "transliteration": "egy_Tnt", "en": "eng_Latn", "de": "deu_Latn"}


class Translator:
    @classmethod
    def from_pretrained(cls, repo_or_dir=None, **kw):
        """Local folder holding the NLLB files, or the Hub repo (its nllb/ subfolder is downloaded)."""
        import os
        from .pipeline import HUB_REPO
        repo_or_dir = repo_or_dir or HUB_REPO
        if not os.path.isdir(repo_or_dir):
            from huggingface_hub import snapshot_download
            repo_or_dir = os.path.join(snapshot_download(repo_or_dir, allow_patterns=["nllb/*"]), "nllb")
        return cls(repo_or_dir, **kw)

    def __init__(self, model_dir, device="auto", dtype="auto", num_beams=4, repetition_penalty=3.0, max_length=4096):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype == "auto":
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.torch, self.device = torch, device
        import os, warnings
        if not os.path.exists(os.path.join(model_dir, "tokenizer.json")):
            warnings.warn("no tokenizer.json in the NLLB folder: transformers 5.x rebuilds the extended NLLB vocabulary "
                          "with every added piece shifted by one id, which garbles transliteration. Use the akhet release "
                          "folder (it ships the exact tokenizer.json) or transformers < 5.")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        import transformers
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}   # kwarg renamed in 4.56
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, **kw).to(device).eval()
        self.gen = dict(num_beams=num_beams, early_stopping=True, max_length=max_length, repetition_penalty=repetition_penalty)
        self.max_length = max_length

    def run(self, text, src, tgt):
        if not text or not text.strip():
            return ""
        self.tok.src_lang = LANG[src]
        enc = self.tok(text, max_length=self.max_length, padding=False, truncation=False, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(LANG[tgt]), **self.gen)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]

    def transliterate(self, blocks, overwrite=False):
        for b in blocks:
            if overwrite or not b.transliteration:
                b.transliteration = self.run(b.hieroglyphs, "hieroglyphs", "transliteration")
        return blocks

    def translate(self, blocks, lang="en", overwrite=False):
        for b in blocks:
            if overwrite or not b.translation.get(lang):
                b.translation[lang] = self.run(b.transliteration or "", "transliteration", lang)
        return blocks
