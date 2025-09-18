from __future__ import annotations
import os, time, json, re
from typing import List, Dict
import numpy as np
import soundfile as sf
import librosa
import torch, torchaudio
from fastapi import FastAPI
from pydantic import BaseModel
from phonemizer import phonemize
from phonemizer.separator import Separator
from ctc_align import build_trellis, backtrack, merge_repeats

APP = FastAPI(title="pron-mvp")
app = APP  # uvicorn main:app 用

USE_PHONEME = os.getenv("USE_PHONEME_BACKEND", "false").lower() == "true"

if not USE_PHONEME:
    BUNDLE = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    ASR_MODEL = BUNDLE.get_model().eval()
    LABELS = BUNDLE.get_labels()
    BLANK_ID = 0  # torchaudio W2V2 CTC の blank
    CHAR_TO_ID = {c: i for i, c in enumerate(LABELS)}
else:
    from phoneme_backend import PhonemeAligner

    PHONEME_MODEL = os.getenv("PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
    ALIGNER = PhonemeAligner(model_id=PHONEME_MODEL)

with open(os.path.join(os.path.dirname(__file__), "presets.json"), "r") as f:
    PRESETS = json.load(f)

SEP = Separator(phone=" ", word="|", syllable="")


class ScoreIn(BaseModel):
    text: str = "This is a test."
    preset: str = "adult"  # elem|junior|high|adult
    audio_path: str | None = None  # 未指定なら /data/samples/sample.wav


@APP.get("/warmup")
def warmup():
    sr = 16000
    x = np.zeros(sr, dtype=np.float32)
    wav = torch.from_numpy(x).unsqueeze(0)
    if not USE_PHONEME:
        with torch.inference_mode():
            emission, _ = ASR_MODEL(wav)  # (T,V)
            emission = emission.log_softmax(-1).squeeze(0)
        return {"ok": True, "backend": "char-ctc", "V": int(emission.size(-1))}
    else:
        em = ALIGNER.emission(wav)
        return {"ok": True, "backend": "phoneme-ctc", "V": int(em.size(-1))}


@APP.post("/score")
def score(inp: ScoreIn):
    t0 = time.time()
    audio_path = inp.audio_path or "/data/samples/sample.wav"
    wav, sr = sf.read(audio_path)
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)

    preset = PRESETS.get(inp.preset, PRESETS["adult"])
    text_words = text_to_words(inp.text)

    if USE_PHONEME:
        # -------- IPA厳密モード --------
        phones_ipa = ALIGNER.align(wav_t, inp.text)  # [{p,start,end,logp}]
        phone_like = phones_to_scores(phones_ipa, preset)

        # 単語ごとの音素配列（phonemizerで正解側の分割を取得）
        ipa_str = phonemize(
            inp.text, language="en-us", backend="espeak", strip=True, separator=SEP
        )
        ipa_words = [
            [p for p in w.strip().split(" ") if p]
            for w in ipa_str.split("|")
            if w.strip()
        ]

        # phone_like を ipa_words の形に合わせて順に切り出し、平均で単語スコア
        words_scored = words_from_phone_sequence(phone_like, ipa_words, text_words)

        overall = int(
            np.clip(
                np.mean([w["score"] for w in words_scored]) if words_scored else 0,
                0,
                100,
            )
        )
        return {
            "overall": overall,
            "preset": inp.preset,
            "words": words_scored,  # ← 追加（単語スコア）
            "chars": [],  # phoneme-CTCでは文字整列は省略
            "phones": phone_like[:1000],
            "diagnostics": {
                "latency_ms": int((time.time() - t0) * 1000),
                "backend": "phoneme-ctc",
            },
        }

    # -------- 高速モード（文字CTC→IPA近似） --------
    with torch.inference_mode():
        emission, _ = ASR_MODEL(wav_t)  # (emission, lengths)
        emission = emission.log_softmax(-1).squeeze(0)  # (T, V)

    ref = normalize_text(inp.text)
    tokens = [CHAR_TO_ID[c] for c in ref if c in CHAR_TO_ID]
    trellis = build_trellis(emission, tokens, BLANK_ID)
    path = backtrack(trellis, emission, tokens, BLANK_ID)
    spans = merge_repeats(path, BLANK_ID)  # [(s,e,tid),...]

    # 文字区間を時間に投影（'|' を含む）
    ratio = wav_t.size(1) / emission.size(0)  # samples_per_frame
    char_segments = []
    for s, e, tid in spans:
        ch = LABELS[tid]
        start = s * ratio / 16000.0
        end = e * ratio / 16000.0
        seg_mean = float(emission[s:e, tid].mean().item())
        char_segments.append({"char": ch, "start": start, "end": end, "logp": seg_mean})

    # テキスト→IPA（単語ごとに分割）
    ipa_str = phonemize(
        inp.text, language="en-us", backend="espeak", strip=True, separator=SEP
    )
    ipa_words = [
        [p for p in w.strip().split(" ") if p] for w in ipa_str.split("|") if w.strip()
    ]

    # 単語区間（char_segments の '|' を境界に）
    word_spans = word_spans_from_chars(char_segments)

    # 近似: 各単語区間を、その単語の音素数で等分して phone 区間を作る→採点
    phone_like = []
    approx_idx = 0
    for i, (ws, we) in enumerate(word_spans):
        phones = ipa_words[i] if i < len(ipa_words) else []
        if not phones:
            continue
        dur = max(1e-6, we - ws)
        step = dur / len(phones)
        approx = []
        cur = ws
        for p in phones:
            approx.append({"p": p, "start": cur, "end": cur + step})
            cur += step
        scored = score_phones_from_chars(emission, approx, char_segments, preset)
        phone_like.extend(scored)

    # 単語スコア（単語区間に重なる phone_like の平均）
    words_scored = aggregate_word_scores_from_time(word_spans, phone_like, text_words)

    overall = int(
        np.clip(
            np.mean([w["score"] for w in words_scored]) if words_scored else 0, 0, 100
        )
    )
    return {
        "overall": overall,
        "preset": inp.preset,
        "words": words_scored,  # ← 追加（単語スコア）
        "chars": char_segments[:1000],
        "phones": phone_like[:1000],
        "diagnostics": {
            "latency_ms": int((time.time() - t0) * 1000),
            "frames": int(emission.size(0)),
            "backend": "char-ctc",
        },
    }


# ---- helpers ----


def normalize_text(s: str) -> str:
    s = s.strip().upper()
    s = re.sub(r"[\.,!?;:'\"-]", "", s)
    s = re.sub(r"\s+", "|", s)
    return s


def text_to_words(s: str) -> List[str]:
    # 表示用の元テキスト単語。記号を外してスペースで分割
    s = re.sub(r"[\.,!?;:'\"-]", "", s.strip())
    ws = [w for w in s.split() if w]
    return ws


def word_spans_from_chars(char_segments: List[Dict]) -> List[tuple]:
    """'|'（空白）で区切って (start,end) の単語区間リストを返す。"""
    spans = []
    cur_start = None
    last_end = None
    for seg in char_segments:
        ch, st, en = seg["char"], seg["start"], seg["end"]
        if ch == "|":
            if cur_start is not None and last_end is not None:
                spans.append((cur_start, last_end))
            cur_start, last_end = None, None
        else:
            if cur_start is None:
                cur_start = st
            last_end = en
    if cur_start is not None and last_end is not None:
        spans.append((cur_start, last_end))
    return spans


def aggregate_word_scores_from_time(
    word_spans: List[tuple], phone_like: List[Dict], text_words: List[str]
) -> List[Dict]:
    out = []
    for i, (ws, we) in enumerate(word_spans):
        ph_in = []
        for ph in phone_like:
            ov = max(0.0, min(we, ph["end"]) - max(ws, ph["start"]))
            if ov > 0:
                ph_in.append(ph["score"])
        score = int(np.clip(np.mean(ph_in) if ph_in else 0, 0, 100))
        out.append(
            {
                "w": text_words[i] if i < len(text_words) else f"word{i+1}",
                "score": score,
            }
        )
    return out


def words_from_phone_sequence(
    phone_like: List[Dict], ipa_words: List[List[str]], text_words: List[str]
) -> List[Dict]:
    """
    phoneme-CTC用：phone_like（順序通り）を ipa_words の各単語の音素数で切り分けて平均。
    """
    out, idx = [], 0
    for i, phones in enumerate(ipa_words):
        n = len(phones)
        slice_ = phone_like[idx : idx + n]
        idx += n
        if not slice_:
            out.append(
                {
                    "w": text_words[i] if i < len(text_words) else f"word{i+1}",
                    "score": 0,
                }
            )
            continue
        score = int(np.clip(np.mean([p["score"] for p in slice_]), 0, 100))
        out.append(
            {
                "w": text_words[i] if i < len(text_words) else f"word{i+1}",
                "score": score,
            }
        )
    return out


def score_phones_from_chars(emission, approx, char_segments, preset):
    tau = preset["tau"]
    beta = preset["beta"]

    def to_score(logp):
        x = (logp - (-tau)) / beta
        return float(100.0 / (1.0 + np.exp(-x)))

    scored = []
    for ph in approx:
        s, e = ph["start"], ph["end"]
        vals = []
        for ch in char_segments:
            ov = max(0.0, min(e, ch["end"]) - max(s, ch["start"]))
            if ov > 0:
                vals.append(ch["logp"])
        lp = float(np.mean(vals)) if vals else -5.0
        scored.append(
            {
                "p": ph["p"],
                "start": s,
                "end": e,
                "score": int(np.clip(to_score(lp), 0, 100)),
            }
        )
    return scored


def phones_to_scores(phones_ipa, preset):
    tau = preset["tau"]
    beta = preset["beta"]

    def to_score(logp):
        x = (logp - (-tau)) / beta
        return float(100.0 / (1.0 + np.exp(-x)))

    out = []
    for ph in phones_ipa:
        out.append(
            {
                "p": ph["p"],
                "start": ph["start"],
                "end": ph["end"],
                "score": int(np.clip(to_score(ph["logp"]), 0, 100)),
            }
        )
    return out
