# 実行手順（10 秒 →<1 秒）

## 0. サンプル音声の配置

- 16kHz/mono・10 秒以内を `data/samples/sample.wav` に置く。
- 変換例：

```bash
ffmpeg -i input.wav -ac 1 -ar 16000 -t 10 data/samples/sample.wav
```

## ASR バックエンドの指定

- 既定で高速モードは Hugging Face 上の `patrickvonplaten/wavlm-base-plus-960h` を読み込みます。
- 別の WavLM-Base+ CTC モデルを利用する場合は `ASR_MODEL_ID`（もしくは互換の `WAVLM_MODEL_ID`）を上書きしてください。
- モデルがプライベート / ゲーテッドの場合は `HF_HOME` にキャッシュをマウントし、`HF_TOKEN`（または `HUGGINGFACEHUB_API_TOKEN`）を環境変数として渡してください。認証に失敗すると API は起動しません。
