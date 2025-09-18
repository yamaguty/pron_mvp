from __future__ import annotations
import torch, numpy as np
from typing import List, Dict
from transformers import AutoProcessor, Wav2Vec2ForCTC
from phonemizer import phonemize
from phonemizer.separator import Separator
from ctc_align import build_trellis, backtrack, merge_repeats


class PhonemeAligner:
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        device: str = "cpu",
    ):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank_id = self.processor.tokenizer.pad_token_id or 0
        self.sep = Separator(phone=" ", word="|", syllable="")

    def _ipa_tokens(self, text: str) -> List[str]:
        s = phonemize(
            text, language="en-us", backend="espeak", separator=self.sep, strip=True
        )
        toks = [t for t in s.split(" ") if t]
        return toks

    def _token_ids(self, ipa_tokens: List[str]) -> List[int]:
        ids = []
        for t in ipa_tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            elif t.replace("ː", "") in self.vocab:
                ids.append(self.vocab[t.replace("ː", "")])
        return ids

    def emission(self, wav_t: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            x = wav_t.squeeze(0).detach().cpu().numpy()
            inputs = self.processor(x, sampling_rate=16000, return_tensors="pt")
            logits = self.model(inputs.input_values).logits
            em = torch.log_softmax(logits, dim=-1).squeeze(0)  # (T, V)
        return em

    def align(self, wav_t: torch.Tensor, text: str) -> List[Dict]:
        ipa_tokens = self._ipa_tokens(text)
        token_ids = self._token_ids(ipa_tokens)
        em = self.emission(wav_t)
        trellis = build_trellis(em, token_ids, self.blank_id)
        path = backtrack(trellis, em, token_ids, self.blank_id)
        segs = merge_repeats(path, self.blank_id)
        ratio = wav_t.size(1) / em.size(0)
        phones = []
        inv_vocab = {v: k for k, v in self.vocab.items()}
        for s, e, tid in segs:
            p = inv_vocab.get(tid, "")
            start = s * ratio / 16000.0
            end = e * ratio / 16000.0
            lp = float(em[s:e, tid].mean().item())
            phones.append({"p": p, "start": start, "end": end, "logp": lp})
        return phones
