# 実行手順（10 秒 →<1 秒）

## 0. サンプル音声の配置

- 16kHz/mono・10 秒以内を `data/samples/sample.wav` に置く。
- 変換例：

```bash
ffmpeg -i input.wav -ac 1 -ar 16000 -t 10 data/samples/sample.wav
```
