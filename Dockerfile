FROM python:3.11-slim


# 必要パッケージ（espeak-ng: phonemizer用、ffmpeg: 変換用）
RUN apt-get update && apt-get install -y --no-install-recommends \
ffmpeg espeak-ng && \
rm -rf /var/lib/apt/lists/*


# PyTorch (CPU) + 依存ライブラリ
# PyTorch (CPU) を明示
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
torch==2.3.1 torchaudio==2.3.1 && \
pip install --no-cache-dir \
fastapi uvicorn[standard] numpy scipy librosa soundfile \
phonemizer transformers==4.44.2 pydantic==2.8.2



# 実行時環境
ENV HF_HOME=/models/hf \
TORCH_HOME=/models/torch \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
ASR_MODEL_ID=patrickvonplaten/wavlm-libri-clean-100h-base-plus \
USE_PHONEME_BACKEND=true \
PHONEME_MODEL=facebook/wav2vec2-lv-60-espeak-cv-ft


WORKDIR /app
COPY app /app

# 環境変数の設定（短い単語対応）
ENV SHORT_WORD_BOOST=1.5
ENV MIN_WORD_LENGTH=8
ENV MIN_PHONE_SCORE=15
ENV WORD_MODE_PENALTY_SCALE=0.5


VOLUME ["/data", "/models"]
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
