# Pron-MVP: 発音評価システム

英語の発音を自動評価するWeb APIシステムです。Transformerベースの音声認識モデルとCTCアライメントを使用して、発音の正確性を0-100点でスコアリングします。

## 機能

- **発音評価**: 音声とテキストを比較して発音の正確性を評価
- **複数モード**: 単語レベル（word）と文レベル（sentence）の評価
- **年齢別プリセット**: 小学生（elem）、中学生（junior）、高校生（high）、大人（adult）の4段階
- **リアルタイム処理**: 高速な音声処理とスコアリング
- **Docker対応**: コンテナ化された環境で簡単にデプロイ可能

## 技術スタック

- **バックエンド**: FastAPI
- **音声処理**: PyTorch, torchaudio, librosa
- **音声認識**: WavLM (Microsoft), wav2vec2 (Facebook)
- **音素処理**: phonemizer, espeak-ng
- **アライメント**: CTC (Connectionist Temporal Classification)
- **コンテナ**: Docker, Docker Compose

## アーキテクチャ

### 2つのバックエンドモード

1. **高速モード（デフォルト）**: WavLM + CTCアライメント
   - 文字レベルのCTCアライメント
   - 高速処理（1秒未満）
   - 近似IPA変換

2. **厳密モード**: wav2vec2 + 音素アライメント
   - 音素レベルの直接アライメント
   - より正確な評価
   - 処理時間が長い

### 評価アルゴリズム

- **CTCアライメント**: 音声とテキストの時間的対応を計算
- **エネルギー重み付け**: 音声の強弱を考慮した評価
- **ノイズ除去**: バンドパスフィルタとプリエンファシス
- **類似度計算**: 認識結果と参照テキストの比較
- **ペナルティシステム**: 構造差、0点音素率、類似度に基づく減点

## セットアップ

### 前提条件

- Docker & Docker Compose
- 8GB以上のRAM（モデル読み込み用）
- インターネット接続（初回モデルダウンロード時）

### インストール

1. リポジトリをクローン
```bash
git clone <repository-url>
cd pron-mvp
```

2. 環境変数を設定
```bash
# Hugging Faceトークン（プライベートモデル用）
export HF_TOKEN="your_huggingface_token"
```

3. Docker Composeで起動
```bash
docker-compose up --build
```

4. APIにアクセス
```
http://localhost:8000
```

### 初回起動時の注意

- 初回起動時はモデルのダウンロードに時間がかかります（数GB）
- モデルは `models/` ディレクトリにキャッシュされます
- 2回目以降は高速に起動します

## API使用方法

### エンドポイント

#### 1. 発音評価 (`POST /score`)

```json
{
  "text": "Hello world",
  "preset": "adult",
  "mode": "sentence",
  "audio_path": "/data/samples/sample.wav"
}
```

**パラメータ:**
- `text`: 評価対象のテキスト
- `preset`: 評価レベル（`elem`, `junior`, `high`, `adult`）
- `mode`: 評価モード（`word`, `sentence`）
- `audio_path`: 音声ファイルのパス（オプション）

**レスポンス:**
```json
{
  "overall": 85,
  "preset": "adult",
  "mode": "sentence",
  "words": [
    {"w": "Hello", "score": 90, "whitelisted": false},
    {"w": "world", "score": 80, "whitelisted": false}
  ],
  "chars": [...],
  "phones": [...],
  "diagnostics": {
    "latency_ms": 1200,
    "mode": "sentence",
    "sim": 0.95,
    "penalty": 0.9
  }
}
```

#### 2. 音声アップロード (`POST /upload`)

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav"
```

#### 3. ヘルスチェック (`GET /warmup`)

```bash
curl http://localhost:8000/warmup
```

## 設定

### 環境変数

| 変数名 | デフォルト | 説明 |
|--------|------------|------|
| `ASR_MODEL_ID` | `patrickvonplaten/wavlm-libri-clean-100h-base-plus` | 音声認識モデル |
| `USE_PHONEME_BACKEND` | `false` | 音素バックエンド使用 |
| `PHONEME_MODEL` | `facebook/wav2vec2-lv-60-espeak-cv-ft` | 音素モデル |
| `NOISE_ROBUST` | `true` | ノイズ除去有効 |
| `ENERGY_WEIGHTING` | `true` | エネルギー重み付け有効 |
| `SIM_THRESH` | `0.25` | 類似度しきい値 |

### プリセット設定

各年齢層に応じた評価パラメータ：

- **elem**: 最も寛容（tau: 0.35, beta: 0.10）
- **junior**: やや寛容（tau: 0.45, beta: 0.10）
- **high**: 標準（tau: 0.55, beta: 0.08）
- **adult**: 厳格（tau: 0.65, beta: 0.08）

## 開発

### プロジェクト構造

```
pron-mvp/
├── app/                    # アプリケーションコード
│   ├── main.py            # FastAPIアプリケーション
│   ├── phoneme_backend.py # 音素バックエンド
│   ├── ctc_align.py       # CTCアライメント
│   └── presets.json       # 評価プリセット
├── data/                   # 音声データ
├── models/                 # 機械学習モデル
├── docker-compose.yml      # Docker Compose設定
├── Dockerfile             # Dockerイメージ定義
└── requirements.txt       # Python依存関係
```

### ローカル開発

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 依存関係インストール
pip install -r requirements.txt

# アプリケーション起動
cd app
uvicorn main:app --reload
```

## デプロイ

### Docker Compose

```bash
# 本番環境用
docker-compose -f docker-compose.prod.yml up -d
```

### Google Cloud Run

```bash
# イメージビルド
docker build -t gcr.io/PROJECT_ID/pron-mvp .

# プッシュ
docker push gcr.io/PROJECT_ID/pron-mvp

# デプロイ
gcloud run deploy pron-mvp \
  --image gcr.io/PROJECT_ID/pron-mvp \
  --platform managed \
  --region asia-northeast1 \
  --memory 8Gi \
  --cpu 2
```

## パフォーマンス

- **処理時間**: 1-3秒（音声長による）
- **メモリ使用量**: 4-8GB（モデルサイズによる）
- **スループット**: 10-50リクエスト/分（ハードウェア依存）

## トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   - Dockerのメモリ制限を8GB以上に設定
   - `OMP_NUM_THREADS`を調整

2. **モデルダウンロードエラー**
   - インターネット接続を確認
   - `HF_TOKEN`を設定（プライベートモデル用）

3. **音声ファイルエラー**
   - サンプリングレート: 16kHz
   - フォーマット: WAV, MP3, FLAC対応

### ログ確認

```bash
# Docker Composeログ
docker-compose logs -f api

# コンテナ内ログ
docker exec -it pron-mvp_api_1 tail -f /app/logs/app.log
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

プルリクエストやイシューの報告を歓迎します。

## 更新履歴

- v1.0.0: 初回リリース
  - 基本的な発音評価機能
  - Docker対応
  - 複数プリセット対応
