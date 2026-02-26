# Changelog (unreleased)

All changes relative to upstream [awni/voxmlx](https://github.com/awni/voxmlx).

## Fix: Use encoder-configured KV window in incremental `encode_step()`

Long streaming sessions in `voxmlx-serve` could degrade semantically after several
sentences when the encoder KV cache window was set to a large fixed value.
The incremental encoder now uses the model's configured encoder sliding window
(`self.encoder.sliding_window`, `750` for Voxtral Mini Realtime) instead of a
hard-coded oversized cache.

**`voxmlx/model.py`**
- `encode_step()`: `RotatingKVCache(self.encoder.sliding_window)` instead of a fixed large value

## Fix: Pin `mistral-common>=1.8.7`

`voxmlx` uses `Tekkenizer.get_special_token()` which was introduced in `mistral-common` [v1.8.7](https://github.com/mistralai/mistral-common/pull/164). Without a version pin, pip can resolve to an older version where this method doesn't exist.

**`pyproject.toml`**
- Pin `mistral-common>=1.8.7`

## Feature: WebSocket server with OpenAI Realtime API

**`voxmlx/server.py`** (new, ~480 lines)
- `StreamingSession` class — extracts all incremental encoder/decoder state from `stream.py` into a reusable object with `feed_audio(audio_f32) -> list[str]` and `finalize() -> list[str]`
- FastAPI WebSocket endpoint at `/v1/realtime` implementing the OpenAI Realtime API protocol:
  - `session.created`/`session.updated` handshake
  - `input_audio_buffer.append` — base64 PCM16 → float32 → `feed_audio()` → `response.audio_transcript.delta` per token
  - `input_audio_buffer.commit` with `final=true` → `finalize()` → flush remaining deltas + `response.audio_transcript.done`
  - Mid-stream EOS from model → auto-sends `done` event, resets session
- `GET /health` endpoint
- CLI: `voxmlx-serve [--model MODEL] [--port PORT] [--host HOST] [--temp TEMP]`

**`voxmlx/__init__.py`**
- Added `--serve`, `--port`, `--host` flags to main CLI (`voxmlx --serve`)

**`pyproject.toml`**
- `[project.optional-dependencies] server = ["fastapi", "uvicorn[standard]"]`
- `voxmlx-serve` entry point → `voxmlx.server:main`
