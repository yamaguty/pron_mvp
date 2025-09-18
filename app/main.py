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
import difflib  # 類似度計算

APP = FastAPI(title="pron-mvp")
app = APP  # uvicorn main:app 用

USE_PHONEME = os.getenv("USE_PHONEME_BACKEND", "false").lower() == "true"
SIM_THRESH = float(os.getenv("REJECT_SIM_THRESH", "0.4"))  # 類似度しきい値（0~1）
NOISE_ROBUST = os.getenv("NOISE_ROBUST", "true").lower() == "false"
TRIM_RATIO = float(os.getenv("TRIM_RATIO", "0.1"))  # トリム平均率 0~0.4 目安0.1
HPF_HZ = float(os.getenv("HPF_HZ", "60"))
LPF_HZ = float(os.getenv("LPF_HZ", "12000"))
PREEMPH = float(os.getenv("PREEMPH", "0.95"))
ENERGY_WEIGHTING = os.getenv("ENERGY_WEIGHTING", "true").lower() == "true"
ENERGY_PCTL = float(os.getenv("ENERGY_PCTL", "0.2"))  # 下位分位点（0~1）
ENERGY_FLOOR_GAIN = float(os.getenv("ENERGY_FLOOR_GAIN", "1.2"))
ENERGY_W_MIN = float(os.getenv("ENERGY_W_MIN", "0.1"))
ENERGY_MASK = os.getenv("ENERGY_MASK", "true").lower() == "true"

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
    if NOISE_ROBUST:
        wav = preprocess_wav_np(wav, sr)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    if NOISE_ROBUST:
        wav_t = bandpass_torch(wav_t, sr=16000, hpf_hz=HPF_HZ, lpf_hz=LPF_HZ)

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

    # 短時間RMSに基づくフレーム重み（教室ガヤ対策）
    weights = None
    if ENERGY_WEIGHTING:
        # emissionの時間長Tに合わせた窓で音声RMSを計算
        T = int(emission.size(0))
        n_samples = int(wav_t.size(1))
        # サンプル/フレーム比
        samples_per_frame = max(1, n_samples // max(1, T))
        x = wav_t.squeeze(0).detach().cpu().numpy()
        rms = []
        for i in range(T):
            s = i * samples_per_frame
            e = min(n_samples, s + samples_per_frame)
            seg = x[s:e]
            if seg.size == 0:
                rms.append(0.0)
            else:
                rms.append(float(np.sqrt(np.mean(seg * seg))))
        rms = np.asarray(rms, dtype=np.float32)
        # 動的ノイズ床（下位分位点）
        p = np.clip(ENERGY_PCTL, 0.0, 1.0)
        floor = np.percentile(rms, p * 100.0)
        thr = max(0.02, float(floor) * ENERGY_FLOOR_GAIN)
        # 正規化重み
        weights = rms - thr
        weights[weights < 0] = 0.0
        if weights.max() > 0:
            weights = weights / (weights.max() + 1e-8)
        weights = np.maximum(weights, ENERGY_W_MIN)
        if ENERGY_MASK:
            # しきい値未満は完全に無視
            weights = np.where(rms >= thr, weights, 0.0)
        weights = torch.from_numpy(weights).to(emission.device).float()  # (T,)

    # 類似度（後段の全体減点に使用）
    hyp_raw = ctc_greedy_decode(emission, LABELS, BLANK_ID)
    hyp = normalize_for_match(hyp_raw)
    tgt = normalize_for_match(inp.text)
    sim = similarity_ratio(hyp, tgt)

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
        if ENERGY_WEIGHTING and weights is not None:
            # 重み付き平均（wは時間のみ依存）
            w = weights[s:e]
            vals = emission[s:e, tid]
            if w.numel() > 0 and float(w.sum().item()) > 0:
                seg_mean = float(
                    (vals * w.unsqueeze(-1)).sum().item() / (w.sum().item())
                )
            else:
                seg_mean = float(robust_segment_mean(vals, trim_ratio=TRIM_RATIO))
        else:
            seg_mean = float(
                robust_segment_mean(emission[s:e, tid], trim_ratio=TRIM_RATIO)
            )
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

    base_overall = float(
        np.clip(
            np.mean([w["score"] for w in words_scored]) if words_scored else 0, 0, 100
        )
    )

    # 構造・不一致に基づく全体減点（部分一致の単語スコアは保持）
    penalty = 1.0
    # 類似度がしきい値未満なら二乗で強めに減点
    if len(hyp) == 0:
        penalty *= 0.3
    elif sim < SIM_THRESH:
        penalty *= max(0.0, (sim / SIM_THRESH)) ** 2
    # 単語数の構造差（文字ベース境界 vs 期待単語数）
    struct_diff = 0.0
    try:
        struct_diff = abs(len(word_spans) - len(text_words)) / max(1, len(text_words))
    except Exception:
        struct_diff = 0.0
    penalty *= max(0.4, 1.0 - 0.8 * struct_diff)
    # phoneレベルで0点が多い場合はさらに減点
    zero_phone_rate = 0.0
    if len(phone_like) > 0:
        zero_phone_rate = sum(1 for p in phone_like if p.get("score", 0) == 0) / len(
            phone_like
        )
        penalty *= max(0.3, 1.0 - 0.8 * zero_phone_rate)
    # 単語の中に0点がある場合の固定減点
    if any(w.get("score", 0) == 0 for w in words_scored):
        penalty *= 0.6

    overall = int(np.clip(base_overall * penalty, 0, 100))
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
            "sim": float(sim),
            "hyp": hyp_raw,
            "penalty": float(penalty),
            "zero_phone_rate": float(zero_phone_rate),
            "struct_diff": float(struct_diff),
            "noise_reduction": bool(NOISE_ROBUST),
            "trim_ratio": float(TRIM_RATIO),
            "preemphasis": float(PREEMPH),
            "bandpass": {"hpf_hz": float(HPF_HZ), "lpf_hz": float(LPF_HZ)},
            "energy_weighting": bool(ENERGY_WEIGHTING),
            "energy_pctl": float(ENERGY_PCTL),
            "energy_floor_gain": float(ENERGY_FLOOR_GAIN),
            "energy_w_min": float(ENERGY_W_MIN),
            "energy_mask": bool(ENERGY_MASK),
        },
    }


# ---- helpers ----


def trimmed_mean(arr, trim_ratio=0.1):
    if not arr:
        return 0.0
    xs = np.array(arr, dtype=np.float32)
    if xs.size < 3:
        return float(xs.mean())
    lo = int(len(xs) * trim_ratio)
    hi = len(xs) - lo
    xs.sort()
    xs = xs[lo:hi] if lo < hi else xs
    return float(xs.mean()) if xs.size > 0 else 0.0


def robust_segment_mean(tensor_vals, trim_ratio=0.1):
    if tensor_vals.numel() == 0:
        return 0.0
    xs = tensor_vals.detach().float().cpu().numpy()
    return trimmed_mean(xs.tolist(), trim_ratio)


def preprocess_wav_np(wav: np.ndarray, sr: int) -> np.ndarray:
    y, _ = librosa.effects.trim(wav, top_db=30)
    y = librosa.effects.preemphasis(y, coef=PREEMPH)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-6:
        y = 0.98 * y / peak
    return y


def bandpass_torch(
    wav_t: torch.Tensor, sr: int, hpf_hz: float, lpf_hz: float
) -> torch.Tensor:
    y = torchaudio.functional.highpass_biquad(wav_t, sample_rate=sr, cutoff_freq=hpf_hz)
    y = torchaudio.functional.lowpass_biquad(y, sample_rate=sr, cutoff_freq=lpf_hz)
    with torch.no_grad():
        peak = y.abs().amax()
        if torch.isfinite(peak) and peak > 1e-6:
            y = y * (0.98 / peak)
    return y


def ctc_greedy_decode(emission, labels, blank_id: int) -> str:
    ids = emission.argmax(dim=-1).tolist()
    out = []
    last = None
    for i in ids:
        if i == blank_id or i == last:
            last = i
            continue
        out.append(labels[i])
        last = i
    return "".join(out)


def normalize_for_match(s: str) -> str:
    s = normalize_text(s)
    return s.replace("|", "")


def similarity_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


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
        score = int(np.clip(trimmed_mean(ph_in, TRIM_RATIO) if ph_in else 0, 0, 100))
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
        score = int(
            np.clip(trimmed_mean([p["score"] for p in slice_], TRIM_RATIO), 0, 100)
        )
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
        lp = float(trimmed_mean(vals, TRIM_RATIO)) if vals else -5.0
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
