## What this fork changes

- **WebSocket server** (`voxmlx-serve`) — OpenAI Realtime API-compatible endpoint for streaming transcription
- **Memory management** — caps MLX's Metal buffer cache at 4 GB (unbounded by default, was hitting 20+ GB), materializes lazy arrays after concat, and clears GPU memory on disconnect
- **DFT matrix caching** — precomputes STFT components instead of rebuilding per `log_mel_spectrogram_step()` call
- **Tighter encoder KV cache** — 8,192 instead of 100,000 to bound active state

See [CHANGES.md](CHANGES.md) for a detailed breakdown.

---

# voxmlx

Realtime speech-to-text with
[Voxtral Mini Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
in [MLX](https://github.com/ml-explore/mlx).

## Install

```bash
pip install voxmlx
```

## Usage

### `voxmlx`

Transcribe audio from a file or stream from the microphone in real-time.

**Stream from microphone:**

```bash
voxmlx
```

**Transcribe a file:**

```bash
voxmlx --audio audio.flac
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--audio` | Path to audio file (omit to stream from mic) | None |
| `--model` | Model path or HuggingFace model ID | `mlx-community/Voxtral-Mini-4B-Realtime-6bit` |
| `--temp` | Sampling temperature (`0` = greedy) | `0.0` |

### `voxmlx-convert`

Convert Voxtral weights to voxmlx/MLX format with optional quantization.

**Basic conversion:**

```bash
voxmlx-convert --mlx-path voxtral-mlx
```

**4-bit quantized conversion:**

```bash
voxmlx-convert -q --mlx-path voxtral-mlx-4bit
```

**Convert and upload to HuggingFace:**

```bash
voxmlx-convert -q --mlx-path voxtral-mlx-4bit --upload-repo username/voxtral-mlx-4bit
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--hf-path` | HuggingFace model ID or local path | `mistralai/Voxtral-Mini-4B-Realtime-2602` |
| `--mlx-path` | Output directory | `mlx_model` |
| `-q`, `--quantize` | Quantize the model | Off |
| `--group-size` | Quantization group size | `64` |
| `--bits` | Bits per weight | `4` |
| `--dtype` | Cast weights (`float16`, `bfloat16`, `float32`) | None |
| `--upload-repo` | HuggingFace repo to upload converted model | None |

### Python API

```python
from voxmlx import transcribe

text = transcribe("audio.flac")
print(text)
```
