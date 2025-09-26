FROM public.ecr.aws/docker/library/python:3.11-slim

# === OS / ランタイム依存 ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg ca-certificates espeak-ng espeak-ng-data && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# PyTorch の CPU 版だけ index-url を強制
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1 torchaudio==2.3.1 && \
    pip install --no-cache-dir -r requirements.txt


# === アプリ & モデルキャッシュを同梱 ===
# app/ 配下に main.py など、models/ 配下に {hf,torch} を置いておく
COPY app/ /app/
COPY models/hf /app/models/hf
COPY models/torch /app/models/torch

# === 実行環境（Cloud Run 向け）===
# HF/Transformers を完全オフラインで動作させる
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/app/models/hf \
    TORCH_HOME=/app/models/torch \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# main.py が読むモデルIDは“通常のリポ名”のままでOK（キャッシュ解決）
# 必要に応じてスナップショット絶対パスにしても構いません
ENV ASR_MODEL_ID=patrickvonplaten/wavlm-libri-clean-100h-base-plus
ENV PHONEME_MODEL=facebook/wav2vec2-lv-60-espeak-cv-ft

# Cloud Run が注入する PORT を使って 0.0.0.0 で待受
ENV PORT=8080
EXPOSE 8080

# 非root 実行（任意だが推奨）
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER 10001

# 起動（readiness を素早く通すため起動を軽く）
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
