from __future__ import annotations
import os, time, json, re
from typing import List, Dict, Optional
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
# デフォルト値をより緩い設定に変更
SIM_THRESH = float(os.getenv("REJECT_SIM_THRESH", "0.4"))  # 類似度しきい値（0~1）
STRUCT_PENALTY_FACTOR = float(
    os.getenv("STRUCT_PENALTY_FACTOR", "0.4")
)  # 構造差ペナルティ係数
ZERO_PHONE_PENALTY_FACTOR = float(
    os.getenv("ZERO_PHONE_PENALTY_FACTOR", "0.4")
)  # ゼロ音素ペナルティ係数
WORD_ZERO_PENALTY = float(os.getenv("WORD_ZERO_PENALTY", "0.7"))  # 単語0点ペナルティ

# 無音除去/前処理用パラメータ
SILENCE_TOP_DB = float(os.getenv("SILENCE_TOP_DB", "25"))  # 無音判定しきい値（dB）
MIN_VOICE_MS = int(os.getenv("MIN_VOICE_MS", "80"))  # セグメント最小長（ms）
PAD_MS = int(os.getenv("PAD_MS", "50"))  # セグメント前後の余白（ms）
PREEMPH_ALPHA = float(os.getenv("PREEMPH_ALPHA", "0.0"))  # 0で無効、例: 0.97

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


class ScoringParams(BaseModel):
    """採点パラメータ"""

    tau: Optional[float] = None  # 音素レベル採点の閾値（低いほど緩い）
    beta: Optional[float] = None  # 音素レベル採点の感度（低いほど緩い）
    sim_thresh: Optional[float] = None  # 類似度しきい値（0~1、低いほど緩い）
    struct_penalty_factor: Optional[float] = (
        None  # 構造差ペナルティ係数（0~1、低いほど緩い）
    )
    zero_phone_penalty_factor: Optional[float] = (
        None  # ゼロ音素ペナルティ係数（0~1、低いほど緩い）
    )
    word_zero_penalty: Optional[float] = None  # 単語0点ペナルティ（0~1、低いほど緩い）


class ScoreIn(BaseModel):
    text: str = "This is a test."
    preset: str = "adult"  # elem|junior|high|adult
    audio_path: str | None = None  # 未指定なら /data/samples/sample.wav
    scoring_params: Optional[ScoringParams] = None  # カスタム採点パラメータ


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
    # ---- 前処理：無音トリム・（任意）プレエンファシス・リサンプリング・モノラル化 ----
    wav, sr = preprocess_wav(
        wav,
        sr,
        target_sr=16000,
        top_db=SILENCE_TOP_DB,
        min_voice_ms=MIN_VOICE_MS,
        pad_ms=PAD_MS,
        preemph_alpha=PREEMPH_ALPHA,
    )
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)

    # 採点パラメータの設定（カスタムパラメータがあれば優先、なければpreset、最後にデフォルト）
    preset = PRESETS.get(inp.preset, PRESETS["adult"]).copy()
    if inp.scoring_params:
        if inp.scoring_params.tau is not None:
            preset["tau"] = inp.scoring_params.tau
        if inp.scoring_params.beta is not None:
            preset["beta"] = inp.scoring_params.beta

    # グローバルパラメータの設定
    sim_thresh = (
        inp.scoring_params.sim_thresh
        if inp.scoring_params and inp.scoring_params.sim_thresh is not None
        else SIM_THRESH
    )
    struct_penalty_factor = (
        inp.scoring_params.struct_penalty_factor
        if inp.scoring_params and inp.scoring_params.struct_penalty_factor is not None
        else STRUCT_PENALTY_FACTOR
    )
    zero_phone_penalty_factor = (
        inp.scoring_params.zero_phone_penalty_factor
        if inp.scoring_params
        and inp.scoring_params.zero_phone_penalty_factor is not None
        else ZERO_PHONE_PENALTY_FACTOR
    )
    word_zero_penalty = (
        inp.scoring_params.word_zero_penalty
        if inp.scoring_params and inp.scoring_params.word_zero_penalty is not None
        else WORD_ZERO_PENALTY
    )

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
    elif sim < sim_thresh:
        penalty *= max(0.0, (sim / sim_thresh)) ** 2
    # 単語数の構造差（文字ベース境界 vs 期待単語数）
    struct_diff = 0.0
    try:
        struct_diff = abs(len(word_spans) - len(text_words)) / max(1, len(text_words))
    except Exception:
        struct_diff = 0.0
    penalty *= max(0.4, 1.0 - struct_penalty_factor * struct_diff)
    # phoneレベルで0点が多い場合はさらに減点
    zero_phone_rate = 0.0
    if len(phone_like) > 0:
        zero_phone_rate = sum(1 for p in phone_like if p.get("score", 0) == 0) / len(
            phone_like
        )
        penalty *= max(0.3, 1.0 - zero_phone_penalty_factor * zero_phone_rate)
    # 単語の中に0点がある場合の固定減点
    if any(w.get("score", 0) == 0 for w in words_scored):
        penalty *= word_zero_penalty

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
            "scoring_params": {
                "sim_thresh": float(sim_thresh),
                "struct_penalty_factor": float(struct_penalty_factor),
                "zero_phone_penalty_factor": float(zero_phone_penalty_factor),
                "word_zero_penalty": float(word_zero_penalty),
            },
        },
    }


# ---- helpers ----


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


def preprocess_wav(
    wav: np.ndarray,
    sr: int,
    target_sr: int = 16000,
    top_db: float = 20.0,
    min_voice_ms: int = 80,
    pad_ms: int = 50,
    preemph_alpha: float = 0.0,
) -> tuple[np.ndarray, int]:
    """無音区間をトリムし、最小長未満の断片を捨て、必要に応じて前後に余白を残す。
    その後リサンプリングとモノラル化を行う。

    - top_db: 無音判定(大きいほど厳しく切る)
    - min_voice_ms: これ未満の短いセグメントを除外
    - pad_ms: 各セグメントの前後に残す余白
    - preemph_alpha: プレエンファシス係数(0で無効)
    """
    # ensure float32 numpy
    wav_np = np.asarray(wav, dtype=np.float32)
    if wav_np.ndim > 1:
        wav_np = np.mean(wav_np, axis=1)

    # リサンプリングを先に統一
    if sr != target_sr:
        wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # プレエンファシス（任意）
    if preemph_alpha and 0.0 < preemph_alpha < 1.0:
        # y[n] = x[n] - a * x[n-1]
        y = np.empty_like(wav_np)
        y[0] = wav_np[0]
        y[1:] = wav_np[1:] - preemph_alpha * wav_np[:-1]
        wav_np = y

    # 無音分割
    intervals = librosa.effects.split(wav_np, top_db=top_db)
    if intervals.size == 0:
        return wav_np, sr

    # パディングサンプル数
    pad = int(pad_ms * sr / 1000)
    min_len = int(min_voice_ms * sr / 1000)

    segments = []
    for beg, end in intervals:
        # 余白を付与
        s = max(0, beg - pad)
        e = min(len(wav_np), end + pad)
        if e - s >= min_len:
            segments.append(wav_np[s:e])

    if not segments:
        return wav_np, sr

    # セグメント連結（間は無音を挿入しない）
    trimmed = np.concatenate(segments, axis=0)

    # クリッピング防止の軽い正規化
    mx = np.max(np.abs(trimmed))
    if mx > 1.0:
        trimmed = trimmed / mx

    return trimmed.astype(np.float32), sr
