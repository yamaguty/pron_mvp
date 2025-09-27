from __future__ import annotations
import os, time, json, re, pprint
from pathlib import Path
from typing import List, Dict, Literal, Optional
import numpy as np
import soundfile as sf
import librosa
import torch, torchaudio
from transformers import AutoModelForCTC, AutoProcessor
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phonemizer import phonemize
from phonemizer.separator import Separator
from ctc_align import build_trellis, backtrack, merge_repeats
import difflib  # 類似度計算

APP = FastAPI(title="pron-mvp")
app = APP  # uvicorn main:app 用

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発中は["*"]でも可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USE_PHONEME = os.getenv("USE_PHONEME_BACKEND", "false").lower() == "true"
SIM_THRESH = float(os.getenv("REJECT_SIM_THRESH", "0.25"))  # 類似度しきい値（0~1）
NOISE_ROBUST = os.getenv("NOISE_ROBUST", "true").lower() == "false"
TRIM_RATIO = float(os.getenv("TRIM_RATIO", "0.05"))  # トリム平均率 0~0.4 目安0.1
HPF_HZ = float(os.getenv("HPF_HZ", "20"))
LPF_HZ = float(os.getenv("LPF_HZ", "7900"))
PREEMPH = float(os.getenv("PREEMPH", "0.95"))
ENERGY_WEIGHTING = os.getenv("ENERGY_WEIGHTING", "true").lower() == "true"
ENERGY_PCTL = float(os.getenv("ENERGY_PCTL", "0.2"))  # 下位分位点（0~1）
ENERGY_FLOOR_GAIN = float(os.getenv("ENERGY_FLOOR_GAIN", "1.2"))
ENERGY_W_MIN = float(os.getenv("ENERGY_W_MIN", "0.1"))
ENERGY_MASK = os.getenv("ENERGY_MASK", "true").lower() == "true"

def _is_cloud_run() -> bool:
    return any(os.getenv(k) for k in ("K_SERVICE", "K_REVISION", "K_CONFIGURATION"))

_DEFAULT_UPLOAD_DIR = "/tmp" if _is_cloud_run() else "/data"
UPLOAD_DATA_DIR = Path(os.getenv("UPLOAD_DATA_DIR", _DEFAULT_UPLOAD_DIR)).expanduser()
_allowed_ips_env = os.getenv("UPLOAD_ALLOWED_IPS", "")
UPLOAD_ALLOWED_IPS = {ip.strip() for ip in _allowed_ips_env.split(",") if ip.strip()}

# 短い単語/フレーズ用の設定
SHORT_WORD_BOOST = float(os.getenv("SHORT_WORD_BOOST", "1.5"))  # 短い単語のスコアブースト
MIN_WORD_LENGTH = int(os.getenv("MIN_WORD_LENGTH", "8"))  # この文字数以下を短いとする
MIN_PHONE_SCORE = float(os.getenv("MIN_PHONE_SCORE", "5"))  # 音素の最低スコア
WORD_MODE_PENALTY_SCALE = float(os.getenv("WORD_MODE_PENALTY_SCALE", "0.35"))  # wordモードでのペナルティ緩和
WORD_MODE_MIN_BASE = float(os.getenv("WORD_MODE_MIN_BASE", "12"))  # wordモード最低保証
WORD_MODE_NO_MATCH_SIM = float(os.getenv("WORD_MODE_NO_MATCH_SIM", "0.1"))  # hypがかすらない判定
SHORT_WORD_MIN_PHONE_SEC = float(os.getenv("SHORT_WORD_MIN_PHONE_SEC", "0.06"))
SHORT_WORD_VOWEL_BONUS = float(os.getenv("SHORT_WORD_VOWEL_BONUS", "1.08"))
SHORT_WORD_STOP_FLOOR = float(os.getenv("SHORT_WORD_STOP_FLOOR", "0.9"))
SHORT_WORD_MIN_STOP_SEC = float(os.getenv("SHORT_WORD_MIN_STOP_SEC", "0.03"))

# Words allowed to miss without triggering zero-score penalties
ZERO_SCORE_WORD_WHITELIST = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "and",
    "in",
    "for",
    "or",
    "but",
}

_VOWELS = {
    "a",
    "e",
    "i",
    "o",
    "u",
    "æ",
    "ɪ",
    "ʊ",
    "ʌ",
    "ɑ",
    "ɔ",
    "ə",
    "ɚ",
    "ɝ",
    "i",
    "u",
    "eɪ",
    "oʊ",
    "aɪ",
    "aʊ",
    "ɔɪ",
    "ɒ",
    "ɛ",
    "ɜ",
}
_SONORANTS = {"l", "r", "m", "n", "ŋ", "j", "w"}
_FRICATIVES = {"s", "z", "ʃ", "ʒ", "f", "v", "θ", "ð", "h"}
_STOPS = {"p", "t", "k", "b", "d", "g", "ʔ"}
_AFFRICATES = {"tʃ", "dʒ"}


def _phone_class(p: str) -> str:
    base = p.replace("ː", "").replace("̩", "")
    if base in _VOWELS or any(base.startswith(v) for v in _VOWELS):
        return "vowel"
    if base in _SONORANTS:
        return "son"
    if base in _FRICATIVES:
        return "fric"
    if base in _AFFRICATES:
        return "aff"
    if base in _STOPS:
        return "stop"
    return "other"


def _dur_weight(p: str) -> float:
    return {
        "vowel": 2.0,
        "son": 1.5,
        "fric": 1.2,
        "aff": 1.1,
        "stop": 0.8,
        "other": 1.0,
    }[_phone_class(p)]


def _short_word_adjust(
    phone: str, base_score: float, logp: float, mode: str, word_length: int
) -> float:
    if not (mode == "word" or word_length <= MIN_WORD_LENGTH):
        return base_score

    cls = _phone_class(phone)
    if cls == "stop":
        floor = MIN_PHONE_SCORE * SHORT_WORD_STOP_FLOOR
        if logp > -8.0:
            base_score = max(base_score, floor)
    if cls == "vowel":
        base_score *= SHORT_WORD_VOWEL_BONUS
    return base_score

_ASR_DEVICE_STR = os.getenv("ASR_DEVICE", "cpu")
try:
    ASR_DEVICE = torch.device(_ASR_DEVICE_STR)
except (TypeError, RuntimeError):
    ASR_DEVICE = torch.device("cpu")

DEFAULT_WAVLM_ASR_MODEL = os.getenv(
    "DEFAULT_WAVLM_ASR_MODEL", "patrickvonplaten/wavlm-libri-clean-100h-base-plus"
)
ASR_MODEL_ID = (
    os.getenv("ASR_MODEL_ID", os.getenv("WAVLM_MODEL_ID", DEFAULT_WAVLM_ASR_MODEL))
    .strip()
    or DEFAULT_WAVLM_ASR_MODEL
)
ASR_BACKEND = "wavlm-ctc"
HF_AUTH_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def token_to_char(token: str) -> str:
    if not token:
        return ""
    if token in {"<pad>", "<s>", "</s>", "<unk>", "<mask>"}:
        return ""
    if token in {"|", "▁", " "}:
        return "|"
    if len(token) == 1 and token.isalpha():
        return token.upper()
    if token.startswith("▁") and len(token) == 2 and token[1].isalpha():
        return token[1].upper()
    return ""

if not USE_PHONEME:
    if not ASR_MODEL_ID:
        raise RuntimeError("ASR_MODEL_ID must be set to a WavLM Base+ model identifier")
    try:
        hf_kwargs = {"token": HF_AUTH_TOKEN} if HF_AUTH_TOKEN else {}
        ASR_PROCESSOR = AutoProcessor.from_pretrained(ASR_MODEL_ID, **hf_kwargs)
        ASR_MODEL = AutoModelForCTC.from_pretrained(ASR_MODEL_ID, **hf_kwargs)
        ASR_MODEL.to(ASR_DEVICE).eval()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load WavLM model '{ASR_MODEL_ID}'. "
            "Ensure the identifier is correct and, if required, provide an HF token."
        ) from exc

    vocab = ASR_PROCESSOR.tokenizer.get_vocab()
    id_to_token = {idx: tok for tok, idx in vocab.items()}
    if not id_to_token:
        raise RuntimeError("Failed to load vocabulary from WavLM model")
    vocab_size = max(id_to_token.keys()) + 1
    LABELS = ["" for _ in range(vocab_size)]
    for idx, token in id_to_token.items():
        LABELS[idx] = token_to_char(token)
    BLANK_ID = ASR_PROCESSOR.tokenizer.pad_token_id
    if BLANK_ID is None:
        BLANK_ID = ASR_PROCESSOR.tokenizer.word_delimiter_token_id
    if BLANK_ID is None:
        raise RuntimeError("Tokenizer must define a pad token to serve as the CTC blank")
    CHAR_TO_ID = {}
    for idx, char in enumerate(LABELS):
        if char and char not in CHAR_TO_ID:
            CHAR_TO_ID[char] = idx
else:
    from phoneme_backend import PhonemeAligner

    PHONEME_MODEL = os.getenv("PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
    ALIGNER = PhonemeAligner(model_id=PHONEME_MODEL)


def compute_asr_emission(wav_t: torch.Tensor) -> torch.Tensor:
    if USE_PHONEME:
        raise RuntimeError("Character backend is disabled; no ASR emission available")
    if ASR_PROCESSOR is None:
        raise RuntimeError("ASR processor is not initialized")
    x = wav_t.squeeze(0).detach().cpu().numpy()
    inputs = ASR_PROCESSOR(x, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(ASR_DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = ASR_MODEL(**inputs).logits
    return torch.log_softmax(logits, dim=-1).squeeze(0).cpu()

with open(os.path.join(os.path.dirname(__file__), "presets.json"), "r") as f:
    PRESETS = json.load(f)

SEP = Separator(phone=" ", word="|", syllable="")


class ScoreIn(BaseModel):
    text: str = "This is a test."
    preset: str = "adult"  # elem|junior|high|adult
    mode: Literal["word", "sentence"] = "sentence"  # word or sentence mode
    audio_path: Optional[str] = None  # 未指定なら /data/samples/sample.wav


def pretty_print_response(label: str, payload: Dict) -> None:
    """Print the response payload in a readable format for debugging."""
    try:
        formatted = json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        formatted = pprint.pformat(payload, indent=2, width=120, compact=False)
    print(f"{label}:\n{formatted}", flush=True)


def get_client_ip(request: Request) -> Optional[str]:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def adjust_preset_for_mode(preset: dict, mode: str, text_length: int) -> dict:
    """モードとテキスト長に応じてプリセットを調整"""
    adjusted = preset.copy()
    
    if mode == "word" or text_length <= MIN_WORD_LENGTH:
        # wordモードまたは短いテキストの場合、より寛容な設定に
        adjusted["tau"] = preset.get("tau", 2.5) * 1.1  # 閾値をさらに下げる
        adjusted["beta"] = preset.get("beta", 0.6) * 1.4  # 曲線をさらに緩やかに
    
    return adjusted


def compute_energy_weights_adaptive(
    wav_t: torch.Tensor, 
    emission: torch.Tensor, 
    mode: str
) -> Optional[torch.Tensor]:
    """モードに応じてエネルギー重み付けを計算"""
    if not ENERGY_WEIGHTING:
        return None
    
    T = int(emission.size(0))
    n_samples = int(wav_t.size(1))
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
    
    # wordモードの場合は異なる閾値設定
    if mode == "word":
        p = max(0.05, ENERGY_PCTL * 0.5)  # より低いパーセンタイル
        floor = np.percentile(rms, p * 100.0)
        thr = max(0.01, float(floor) * 1.0)  # より低い閾値
        min_w = 0.3  # より高い最小重み
    else:
        p = np.clip(ENERGY_PCTL, 0.0, 1.0)
        floor = np.percentile(rms, p * 100.0)
        thr = max(0.02, float(floor) * ENERGY_FLOOR_GAIN)
        min_w = ENERGY_W_MIN
    
    weights = rms - thr
    weights[weights < 0] = 0.0
    
    if weights.max() > 0:
        weights = weights / (weights.max() + 1e-8)
    
    weights = np.maximum(weights, min_w)
    
    if ENERGY_MASK and mode == "sentence":
        # sentenceモードの場合のみマスクを適用
        weights = np.where(rms >= thr, weights, 0.0)
    
    return torch.from_numpy(weights).float()


def preprocess_wav_adaptive(
    wav: np.ndarray, 
    sr: int, 
    mode: str
) -> np.ndarray:
    """モードに応じた前処理"""
    if mode == "word":
        # wordモードの場合はより緩いトリミング
        y, _ = librosa.effects.trim(wav, top_db=28)
        preemph_coef = 0.9  # より弱いプリエンファシス
    else:
        y, _ = librosa.effects.trim(wav, top_db=20)
        preemph_coef = PREEMPH
    
    # トリムで空 or 短すぎる場合は元の波形を使う
    min_duration = 0.2 if mode == "word" else 0.3
    if y.size == 0 or len(y) < int(min_duration * sr):
        y = wav
    
    # プリエンファシス
    y = librosa.effects.preemphasis(y, coef=preemph_coef)
    
    # 正規化
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-6:
        y = 0.98 * y / peak
    
    return y.astype(np.float32)


def adaptive_bandpass(
    wav_t: torch.Tensor, 
    sr: int, 
    mode: str
) -> torch.Tensor:
    """モードに応じたバンドパスフィルタ"""
    if mode == "word":
        # wordモードの場合はより広い周波数帯を保持
        hpf = 10
        lpf = 8000
    else:
        hpf = HPF_HZ
        lpf = LPF_HZ
    
    y = torchaudio.functional.highpass_biquad(wav_t, sample_rate=sr, cutoff_freq=hpf)
    y = torchaudio.functional.lowpass_biquad(y, sample_rate=sr, cutoff_freq=lpf)
    
    # NaN/Inf ガード
    if not torch.isfinite(y).all():
        print("WARNING: NaN detected in bandpass, applying nan_to_num")
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    return y


@APP.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    client_ip = get_client_ip(request)
    # if UPLOAD_ALLOWED_IPS and (client_ip is None or client_ip not in UPLOAD_ALLOWED_IPS):
    #     raise HTTPException(status_code=403, detail="Uploads are not permitted from this IP address.")

    safe_name = Path(file.filename or "").name
    if not safe_name:
        await file.close()
        raise HTTPException(status_code=400, detail="A valid filename is required.")

    try:
        UPLOAD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        await file.close()
        raise HTTPException(status_code=500, detail=f"Failed to prepare upload directory: {exc}")

    destination = UPLOAD_DATA_DIR / safe_name
    bytes_written = 0

    try:
        with destination.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
                bytes_written += len(chunk)
    except OSError as exc:
        await file.close()
        raise HTTPException(status_code=500, detail=f"Failed to write upload: {exc}")
    finally:
        await file.close()

    return {
        "filename": safe_name,
        "bytes_written": bytes_written,
        "path": str(destination),
    }


@APP.get("/warmup")
def warmup():
    sr = 16000
    x = np.zeros(sr, dtype=np.float32)
    wav = torch.from_numpy(x).unsqueeze(0)
    if not USE_PHONEME:
        emission = compute_asr_emission(wav)
        return {"ok": True, "backend": ASR_BACKEND, "V": int(emission.size(-1))}
    else:
        em = ALIGNER.emission(wav)
        return {"ok": True, "backend": "phoneme-ctc", "V": int(em.size(-1))}


@APP.post("/score")
def score(inp: ScoreIn):
    t0 = time.time()
    audio_path = inp.audio_path or "/data/samples/sample.wav"
    mode = inp.mode  # word or sentence
    
    # デバッグ情報
    print(f"Processing in {mode} mode for text: '{inp.text}'", flush=True)
    
    wav, sr = sf.read(audio_path)
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    
    if NOISE_ROBUST:
        wav = preprocess_wav_adaptive(wav, sr, mode)
        print(f"DEBUG preprocess ({mode} mode): len=", len(wav), 
            "dtype=", wav.dtype, 
            "RMS=", float(np.sqrt(np.mean(wav**2))) if wav.size else 0.0, flush=True)

    wav_t = torch.from_numpy(wav).float().unsqueeze(0)

    if NOISE_ROBUST:
        wav_t = adaptive_bandpass(wav_t, sr=16000, mode=mode)
        print(f"DEBUG bandpass ({mode} mode): shape=", tuple(wav_t.shape),
            "RMS=", float(wav_t.pow(2).mean().sqrt().item()),
            "min=", float(wav_t.min().item()),
            "max=", float(wav_t.max().item()), flush=True)

    # テキスト長に基づくプリセット調整
    text_length = len(inp.text.replace(" ", ""))
    preset = PRESETS.get(inp.preset, PRESETS["adult"])
    preset = adjust_preset_for_mode(preset, mode, text_length)
    
    text_words = text_to_words(inp.text)

    if USE_PHONEME:
        # -------- IPA厳密モード --------
        phones_ipa = ALIGNER.align(wav_t, inp.text)  # [{p,start,end,logp}]
        phone_like = phones_to_scores(phones_ipa, preset, mode)

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

        scores_for_mean = [
            w["score"]
            for w in words_scored
            if not (w["score"] == 0 and w.get("whitelisted"))
        ]
        
        # モードに応じた全体スコア計算
        base_overall = float(
            np.clip(np.mean(scores_for_mean) if scores_for_mean else 0, 0, 100)
        )
        
        if mode == "word" and base_overall < 20 and text_length <= MIN_WORD_LENGTH:
            base_overall = 20  # wordモードの最低保証
        
        overall = int(base_overall)
        
        result = {
            "overall": overall,
            "preset": inp.preset,
            "mode": mode,
            "words": words_scored,
            "chars": [],  # phoneme-CTCでは文字整列は省略
            "phones": phone_like[:1000],
            "diagnostics": {
                "latency_ms": int((time.time() - t0) * 1000),
                "backend": "phoneme-ctc",
                "mode": mode,
            },
        }

        pretty_print_response("/score result", result)
        return result

    # -------- 高速モード（ASR CTC→IPA近似） --------
    emission = compute_asr_emission(wav_t)

    # エネルギー重み付け（モード対応）
    weights = compute_energy_weights_adaptive(wav_t, emission, mode)

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
        if tid >= len(LABELS):
            continue
        ch = LABELS[tid]
        if not ch:
            continue
        start = s * ratio / 16000.0
        end = e * ratio / 16000.0
        if weights is not None:
            # 重み付き平均
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

    # CTCの '|' が欠落して単語数が一致しない場合のフォールバック分割
    if mode == "word" and len(word_spans) != len(text_words):
        if char_segments:
            t0 = char_segments[0]["start"]
            t1 = char_segments[-1]["end"]
        else:
            t0 = 0.0
            t1 = float(wav_t.size(1) / 16000.0)
        t1 = max(t1, t0)

        parts = [max(1, len(ws)) for ws in ipa_words]
        if len(parts) < len(text_words):
            parts.extend([1] * (len(text_words) - len(parts)))
        elif len(parts) > len(text_words):
            parts = parts[: len(text_words)]
        if not parts:
            parts = [1] * max(1, len(text_words))

        total = float(sum(parts)) or 1.0
        cuts = [t0]
        acc = t0
        span = t1 - t0
        for n in parts[:-1]:
            acc += span * (n / total)
            cuts.append(acc)
        cuts.append(t1)
        word_spans = [(cuts[i], cuts[i + 1]) for i in range(len(parts))]

    # 近似: 各単語区間を、その単語の音素数で等分してphone区間を作る→採点
    phone_like = []
    for i, (ws, we) in enumerate(word_spans):
        phones = ipa_words[i] if i < len(ipa_words) else []
        if not phones:
            continue
        word_label = text_words[i] if i < len(text_words) else f"word{i+1}"
        is_short_word = mode == "word" or len(word_label) <= MIN_WORD_LENGTH
        dur = max(1e-6, we - ws)
        approx = []

        if is_short_word:
            weights = np.array([_dur_weight(p) for p in phones], dtype=np.float32)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            weights = weights / (weights.sum() + 1e-8)
            target = weights * dur

            min_durs = np.full(len(phones), SHORT_WORD_MIN_PHONE_SEC, dtype=np.float32)
            for idx, phone in enumerate(phones):
                if _phone_class(phone) == "stop":
                    min_durs[idx] = SHORT_WORD_MIN_STOP_SEC

            min_total = float(min_durs.sum()) if min_durs.size else 0.0
            if dur <= min_total and min_total > 0:
                target = min_durs.copy()
            else:
                need = np.maximum(0.0, min_durs - target)
                if need.sum() > 0:
                    extra = float(need.sum())
                    can_give = np.maximum(0.0, target - min_durs)
                    if can_give.sum() > 1e-8:
                        give_ratio = min(1.0, extra / (can_give.sum() + 1e-8))
                        target = target - can_give * give_ratio + need
                    else:
                        target = min_durs.copy()
            total = float(target.sum())
            if total > 0:
                scale = dur / (total + 1e-8)
                target = target * scale

            seg_lengths = target.astype(np.float32)
        else:
            step = float(dur / len(phones))
            seg_lengths = np.full(len(phones), step, dtype=np.float32)

        cur = ws
        for p, seglen in zip(phones, seg_lengths):
            seg_dur = float(seglen)
            approx.append({"p": p, "start": cur, "end": cur + seg_dur})
            cur += seg_dur

        if approx:
            approx[-1]["end"] = we

        scored = score_phones_from_chars(
            emission, approx, char_segments, preset, mode, word_label
        )

        is_whitelisted = is_whitelisted_zero_word(word_label)
        for sc in scored:
            sc["word"] = word_label
            sc["whitelisted"] = is_whitelisted
        phone_like.extend(scored)

    # 単語スコア（単語区間に重なる phone_like の平均）
    words_scored = aggregate_word_scores_from_time(word_spans, phone_like, text_words)

    scores_for_mean = [
        w["score"]
        for w in words_scored
        if not (w["score"] == 0 and w.get("whitelisted"))
    ]
    base_overall = float(
        np.clip(np.mean(scores_for_mean) if scores_for_mean else 0, 0, 100)
    )

    # モードに応じたペナルティ計算
    penalty = 1.0
    
    if mode == "word":
        # wordモードの場合はペナルティを緩和
        filtered_phones = [p for p in phone_like if not p.get("whitelisted")]
        has_phone_partial = any(p.get("score", 0) > 0 for p in filtered_phones)

        if len(hyp) == 0 and not has_phone_partial:
            penalty = 0.0
        elif sim < SIM_THRESH and not has_phone_partial:
            # 類似度が低い場合は線形にペナルティ（音素部分点がなければゼロへ収束）
            denom = SIM_THRESH if SIM_THRESH > 1e-6 else 1.0
            similarity_scale = max(0.0, min(1.0, sim / denom))
            penalty *= similarity_scale
        
        # 構造差のペナルティも緩和
        struct_diff = 0.0
        try:
            struct_diff = abs(len(word_spans) - len(text_words)) / max(1, len(text_words))
        except Exception:
            struct_diff = 0.0
        penalty *= max(0.6, 1.0 - 0.5 * struct_diff)
        
        # 0点phoneの影響も緩和
        zero_phone_rate = 0.0
        if filtered_phones:
            zero_phone_rate = sum(1 for p in filtered_phones if p.get("score", 0) == 0) / len(
                filtered_phones
            )
            penalty *= max(0.5, 1.0 - 0.5 * zero_phone_rate)
        
        # 0点単語の影響も緩和
        if any(
            w.get("score", 0) == 0 and not w.get("whitelisted") for w in words_scored
        ):
            penalty *= 0.85

        # hyp がかすりもしない場合は 0 点確定（音素部分点がある場合は除く）
        no_hyp_match = len(hyp) == 0 or sim < WORD_MODE_NO_MATCH_SIM
        if no_hyp_match and not has_phone_partial:
            base_overall = 0.0
            penalty = 0.0

        # ペナルティを許容範囲に制限
        penalty = max(0.0, min(penalty, 1.0))
        if penalty > 0.0:
            penalty = penalty ** WORD_MODE_PENALTY_SCALE
        
    else:
        # sentenceモード（既存のロジック）
        if len(hyp) == 0:
            penalty *= 0.3
        elif sim < SIM_THRESH:
            penalty *= max(0.0, (sim / SIM_THRESH)) ** 2
        
        # 構造差
        struct_diff = 0.0
        try:
            struct_diff = abs(len(word_spans) - len(text_words)) / max(1, len(text_words))
        except Exception:
            struct_diff = 0.0
        penalty *= max(0.4, 1.0 - 0.8 * struct_diff)
        
        # phoneレベルで0点が多い場合
        zero_phone_rate = 0.0
        filtered_phones = [p for p in phone_like if not p.get("whitelisted")]
        if filtered_phones:
            zero_phone_rate = sum(1 for p in filtered_phones if p.get("score", 0) == 0) / len(
                filtered_phones
            )
            penalty *= max(0.3, 1.0 - 0.8 * zero_phone_rate)
        
        # 単語の中に0点がある場合
        if any(
            w.get("score", 0) == 0 and not w.get("whitelisted") for w in words_scored
        ):
            penalty *= 0.6

    # wordモードで短いテキストの場合、最低スコアを保証
    if (
        mode == "word"
        and text_length <= MIN_WORD_LENGTH
        and base_overall < WORD_MODE_MIN_BASE
        and penalty > 0.5
    ):
        base_overall = WORD_MODE_MIN_BASE

    overall = int(np.clip(base_overall * penalty, 0, 100))
    
    result = {
        "overall": overall,
        "preset": inp.preset,
        "mode": mode,
        "words": words_scored,
        "chars": char_segments[:1000],
        "phones": phone_like[:1000],
        "diagnostics": {
            "latency_ms": int((time.time() - t0) * 1000),
            "frames": int(emission.size(0)),
            "backend": ASR_BACKEND,
            "model_id": ASR_MODEL_ID,
            "mode": mode,
            "sim": float(sim),
            "hyp": hyp_raw,
            "penalty": float(penalty),
            "zero_phone_rate": float(zero_phone_rate) if 'zero_phone_rate' in locals() else 0.0,
            "struct_diff": float(struct_diff) if 'struct_diff' in locals() else 0.0,
            "noise_reduction": bool(NOISE_ROBUST),
            "trim_ratio": float(TRIM_RATIO),
            "preemphasis": float(PREEMPH),
            "bandpass": {"hpf_hz": float(hpf) if 'hpf' in locals() else HPF_HZ, 
                        "lpf_hz": float(lpf) if 'lpf' in locals() else LPF_HZ},
            "energy_weighting": bool(ENERGY_WEIGHTING),
            "energy_pctl": float(ENERGY_PCTL),
            "energy_floor_gain": float(ENERGY_FLOOR_GAIN),
            "energy_w_min": float(ENERGY_W_MIN),
            "energy_mask": bool(ENERGY_MASK),
        },
    }

    pretty_print_response("/score result", result)
    return result


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
    y, _ = librosa.effects.trim(wav, top_db=20)

    # トリムで空 or 短すぎる場合は元の波形を使う
    if y.size == 0 or len(y) < int(0.3 * sr):
        y = wav

    # プリエンファシス
    y = librosa.effects.preemphasis(y, coef=PREEMPH)

    # 正規化
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 1e-6:
        y = 0.98 * y / peak
    return y.astype(np.float32)


def bandpass_torch(
    wav_t: torch.Tensor, sr: int, hpf_hz: float, lpf_hz: float
) -> torch.Tensor:
    y = torchaudio.functional.highpass_biquad(wav_t, sample_rate=sr, cutoff_freq=hpf_hz)
    y = torchaudio.functional.lowpass_biquad(y, sample_rate=sr, cutoff_freq=lpf_hz)

    # NaN/Inf ガード
    if not torch.isfinite(y).all():
        print("WARNING: NaN detected in bandpass, applying nan_to_num")
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y


def ctc_greedy_decode(emission, labels, blank_id: int) -> str:
    ids = emission.argmax(dim=-1).tolist()
    out = []
    last = None
    for i in ids:
        if i == blank_id or i == last:
            last = i
            continue
        if i >= len(labels):
            last = i
            continue
        ch = labels[i]
        if not ch:
            last = i
            continue
        out.append(ch)
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


def is_whitelisted_zero_word(word: str) -> bool:
    return word.strip().lower() in ZERO_SCORE_WORD_WHITELIST


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
        word_label = text_words[i] if i < len(text_words) else f"word{i+1}"
        out.append(
            {
                "w": word_label,
                "score": score,
                "whitelisted": is_whitelisted_zero_word(word_label),
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
        word_label = text_words[i] if i < len(text_words) else f"word{i+1}"
        n = len(phones)
        slice_ = phone_like[idx : idx + n]
        idx += n
        if not slice_:
            out.append(
                {
                    "w": word_label,
                    "score": 0,
                    "whitelisted": is_whitelisted_zero_word(word_label),
                }
            )
            continue
        score = int(
            np.clip(trimmed_mean([p["score"] for p in slice_], TRIM_RATIO), 0, 100)
        )
        out.append(
            {
                "w": word_label,
                "score": score,
                "whitelisted": is_whitelisted_zero_word(word_label),
            }
        )
    return out


def score_phones_from_chars(
    emission, approx, char_segments, preset, mode="sentence", word_label=""
):
    """モードと単語に応じてスコアを計算"""
    # 単語の長さに基づく調整
    word_length = len(word_label) if word_label else 10
    
    tau = preset["tau"]
    beta = preset["beta"]
    
    # wordモードまたは短い単語の場合の調整
    if mode == "word" or word_length <= MIN_WORD_LENGTH:
        tau = tau * 1.3  # 閾値をさらに緩める
        beta = beta * 1.2  # 曲線をさらに緩やかに

    def to_score(ph, logp):
        x = (logp - (-tau)) / beta
        base_score = float(100.0 / (1.0 + np.exp(-x)))

        if mode == "word" or word_length <= MIN_WORD_LENGTH:
            base_score = base_score * SHORT_WORD_BOOST

        if mode == "word":
            if logp > -5.0 and base_score < MIN_PHONE_SCORE:
                base_score = MIN_PHONE_SCORE
        else:
            if logp > -5.0 and base_score < (MIN_PHONE_SCORE * 0.5):
                base_score = MIN_PHONE_SCORE * 0.5

        base_score = _short_word_adjust(ph["p"], base_score, logp, mode, word_length)
        return base_score


    scored = []
    for ph in approx:
        s, e = ph["start"], ph["end"]
        vals = []
        for ch in char_segments:
            ov = max(0.0, min(e, ch["end"]) - max(s, ch["start"]))
            if ov > 0:
                vals.append(ch["logp"])
        
        # 値が取得できない場合のデフォルト値を調整
        if not vals:
            # wordモードの場合はより寛容なデフォルト値
            lp = -3.0 if mode == "word" else -5.0
        else:
            lp = float(trimmed_mean(vals, TRIM_RATIO))
        
        score_val = int(np.clip(to_score(ph, lp), 0, 100))
        scored.append(
            {
                "p": ph["p"],
                "start": s,
                "end": e,
                "score": score_val,
            }
        )
    return scored


def phones_to_scores(phones_ipa, preset, mode="sentence"):
    """phoneme-CTC用のスコア計算（モード対応）"""
    tau = preset["tau"]
    beta = preset["beta"]
    
    # wordモードの調整
    if mode == "word":
        tau = tau * 1.3
        beta = beta * 1.2

    def to_score(ph):
        dur = max(1e-6, ph["end"] - ph["start"])
        tt, bb = tau, beta
        if mode == "word" or dur < 0.06:
            tt *= 1.2
            bb *= 1.2

        x = (ph["logp"] - (-tt)) / bb
        base_score = float(100.0 / (1.0 + np.exp(-x)))

        if mode == "word":
            base_score = base_score * SHORT_WORD_BOOST

        adjust_length = MIN_WORD_LENGTH if mode == "word" else MIN_WORD_LENGTH + 1
        base_score = _short_word_adjust(ph["p"], base_score, ph["logp"], mode, adjust_length)

        if mode == "word" and ph["logp"] > -5.0 and base_score < MIN_PHONE_SCORE:
            base_score = MIN_PHONE_SCORE

        return base_score

    out = []
    for ph in phones_ipa:
        out.append(
            {
                "p": ph["p"],
                "start": ph["start"],
                "end": ph["end"],
                "score": int(np.clip(to_score(ph), 0, 100)),
            }
        )
    return out
