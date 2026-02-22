"""
WebSocket server for voxmlx that speaks the OpenAI Realtime API protocol.

Start with:  voxmlx-serve [--model MODEL] [--port PORT] [--temp TEMP]
Or:          python -m voxmlx.server
"""

import argparse
import base64
import json
import logging
import time

import mlx.core as mx
import numpy as np
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from . import _build_prompt_tokens, load_model
from .audio import SAMPLES_PER_TOKEN, log_mel_spectrogram_step
from .cache import RotatingKVCache

logger = logging.getLogger("voxmlx.server")

N_LEFT_PAD_TOKENS = 32
N_RIGHT_PAD_TOKENS = 17


class StreamingSession:
    """Encapsulates all incremental encoder/decoder state for one utterance."""

    def __init__(self, model, sp, temperature=0.0):
        self.model = model
        self.sp = sp
        self.temperature = temperature

        prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)
        self.prefix_len = len(prompt_tokens)
        self.eos_token_id = sp.eos_id

        self.t_cond = model.time_embedding(
            mx.array([n_delay_tokens], dtype=mx.float32)
        )
        mx.eval(self.t_cond)

        prompt_ids = mx.array([prompt_tokens])
        self.text_embeds = model.language_model.embed(prompt_ids)[0]
        mx.eval(self.text_embeds)

        self.n_layers = len(model.language_model.layers)
        self.sliding_window = 8192

        self._reset_state()

    def _reset_state(self):
        """Clear all incremental state for a new utterance."""
        # Decoder state
        self.cache = None
        self.y = None

        # Encoder state
        self.audio_tail = None
        self.conv1_tail = None
        self.conv2_tail = None
        self.encoder_cache = None
        self.ds_buf = None

        # Buffers and counters
        self.pending_audio = np.zeros(0, dtype=np.float32)
        self.audio_embeds = None
        self.n_audio_samples_fed = 0
        self.n_total_decoded = 0
        self.first_cycle = True
        self.prefilled = False
        self.full_text = ""

    def reset(self):
        """Public reset — clears state for the next utterance."""
        self._reset_state()

    def _sample(self, logits):
        if self.temperature <= 0:
            return mx.argmax(logits[0, -1:], axis=-1).squeeze()
        return mx.random.categorical(logits[0, -1:] / self.temperature).squeeze()

    def _decode_steps(self, embeds, n_to_decode):
        """Decode n_to_decode positions. Returns (n_consumed, hit_eos, tokens)."""
        tokens = []
        for i in range(n_to_decode):
            token_embed = self.model.language_model.embed(
                self.y.reshape(1, 1)
            )[0, 0]
            step_embed = (embeds[i] + token_embed)[None, None, :]
            logits = self.model.decode(
                step_embed, self.t_cond, mask=None, cache=self.cache
            )
            next_y = self._sample(logits)
            mx.async_eval(next_y)

            token_id = self.y.item()
            if token_id == self.eos_token_id:
                self.cache = None
                self.y = None
                return i, True, tokens

            text = self.sp.decode([token_id], special_token_policy=SpecialTokenPolicy.IGNORE)
            tokens.append(text)
            self.full_text += text

            if i > 0 and i % 256 == 0:
                mx.clear_cache()

            self.y = next_y

        return n_to_decode, False, tokens

    def _encode_audio(self):
        """Encode pending audio into embeddings. Returns True if new embeds produced."""
        if self.first_cycle and len(self.pending_audio) >= SAMPLES_PER_TOKEN:
            left_pad = np.zeros(
                N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32
            )
            n_feed = (
                len(self.pending_audio) // SAMPLES_PER_TOKEN
            ) * SAMPLES_PER_TOKEN
            chunk = np.concatenate([left_pad, self.pending_audio[:n_feed]])
            self.pending_audio = self.pending_audio[n_feed:]
            self.n_audio_samples_fed += n_feed

            mel, self.audio_tail = log_mel_spectrogram_step(
                chunk, self.audio_tail
            )
            new_embeds, self.conv1_tail, self.conv2_tail, self.encoder_cache, self.ds_buf = (
                self.model.encode_step(
                    mel,
                    self.conv1_tail,
                    self.conv2_tail,
                    self.encoder_cache,
                    self.ds_buf,
                )
            )
            if new_embeds is not None:
                mx.eval(new_embeds)
                self.audio_embeds = new_embeds
            self.first_cycle = False
            return True

        elif not self.first_cycle and len(self.pending_audio) >= SAMPLES_PER_TOKEN:
            n_feed = (
                len(self.pending_audio) // SAMPLES_PER_TOKEN
            ) * SAMPLES_PER_TOKEN
            chunk = self.pending_audio[:n_feed]
            self.pending_audio = self.pending_audio[n_feed:]
            self.n_audio_samples_fed += n_feed

            mel, self.audio_tail = log_mel_spectrogram_step(
                chunk, self.audio_tail
            )
            new_embeds, self.conv1_tail, self.conv2_tail, self.encoder_cache, self.ds_buf = (
                self.model.encode_step(
                    mel,
                    self.conv1_tail,
                    self.conv2_tail,
                    self.encoder_cache,
                    self.ds_buf,
                )
            )
            if new_embeds is not None:
                mx.eval(new_embeds)
                if self.audio_embeds is not None:
                    self.audio_embeds = mx.concatenate(
                        [self.audio_embeds, new_embeds]
                    )
                else:
                    self.audio_embeds = new_embeds
            return True

        return False

    def _try_prefill(self):
        """Attempt prefill if we have enough embeddings. Returns True if prefilled."""
        if self.prefilled or self.audio_embeds is None:
            return False
        if self.n_total_decoded + self.audio_embeds.shape[0] < self.prefix_len:
            return False

        self.cache = [
            RotatingKVCache(self.sliding_window) for _ in range(self.n_layers)
        ]

        prefix_embeds = self.text_embeds + self.audio_embeds[: self.prefix_len]
        prefix_embeds = prefix_embeds[None, :, :]

        logits = self.model.decode(
            prefix_embeds, self.t_cond, "causal", self.cache
        )
        mx.eval(logits, *[x for c in self.cache for x in (c.keys, c.values)])

        self.y = self._sample(logits)
        mx.async_eval(self.y)

        self.audio_embeds = self.audio_embeds[self.prefix_len :]
        self.n_total_decoded = self.prefix_len
        self.prefilled = True
        return True

    def _decode_available(self):
        """Decode all available embeddings. Returns list of decoded token texts."""
        if self.audio_embeds is None:
            return []

        safe_total = (
            N_LEFT_PAD_TOKENS + self.n_audio_samples_fed // SAMPLES_PER_TOKEN
        )
        n_decodable = min(
            self.audio_embeds.shape[0], safe_total - self.n_total_decoded
        )
        if n_decodable <= 0:
            return []

        n_consumed, hit_eos, tokens = self._decode_steps(
            self.audio_embeds, n_decodable
        )
        self.n_total_decoded += n_consumed

        if self.audio_embeds.shape[0] > n_consumed:
            self.audio_embeds = self.audio_embeds[n_consumed:]
        else:
            self.audio_embeds = None

        if hit_eos:
            full = self.full_text
            self._reset_state()
            return tokens, full  # signal EOS with tuple

        return tokens

    def feed_audio(self, audio_f32: np.ndarray):
        """Feed audio samples and return decoded tokens.

        Returns list of token strings, or None if EOS was hit
        (in which case .eos_text contains the full utterance text).
        """
        self.pending_audio = np.append(self.pending_audio, audio_f32)
        self.eos_text = None

        all_tokens = []

        # Encode all available audio
        while len(self.pending_audio) >= SAMPLES_PER_TOKEN:
            self._encode_audio()

            if not self.prefilled:
                self._try_prefill()

            if self.prefilled:
                result = self._decode_available()
                if isinstance(result, tuple):
                    # EOS hit: (tokens_before_eos, full_text)
                    all_tokens.extend(result[0])
                    self.eos_text = result[1]
                    return all_tokens
                all_tokens.extend(result)

        return all_tokens

    def finalize(self):
        """Flush remaining audio with right padding. Returns remaining token texts.

        Returns list of tokens. If EOS is hit, .eos_text is set.
        """
        self.eos_text = None
        was_first_cycle = self.first_cycle

        # Flush any remaining pending audio + right padding
        right_pad = np.zeros(
            N_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32
        )
        flush_chunk = np.concatenate([self.pending_audio, right_pad])
        self.pending_audio = np.zeros(0, dtype=np.float32)

        if was_first_cycle:
            # Need left pad for first cycle
            left_pad = np.zeros(
                N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32
            )
            flush_chunk = np.concatenate([left_pad, flush_chunk])

        n_feed = (len(flush_chunk) // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
        if n_feed == 0:
            return []

        chunk = flush_chunk[:n_feed]
        # Only count real audio samples, not padding
        pad_samples = (
            N_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN
            + (N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN if was_first_cycle else 0)
        )
        self.n_audio_samples_fed += n_feed - pad_samples

        mel, self.audio_tail = log_mel_spectrogram_step(chunk, self.audio_tail)
        new_embeds, self.conv1_tail, self.conv2_tail, self.encoder_cache, self.ds_buf = (
            self.model.encode_step(
                mel,
                self.conv1_tail,
                self.conv2_tail,
                self.encoder_cache,
                self.ds_buf,
            )
        )
        if new_embeds is not None:
            mx.eval(new_embeds)
            if self.audio_embeds is not None:
                self.audio_embeds = mx.concatenate(
                    [self.audio_embeds, new_embeds]
                )
            else:
                self.audio_embeds = new_embeds

        if was_first_cycle:
            self.first_cycle = False

        # Prefill if needed
        if not self.prefilled:
            self._try_prefill()

        # Decode everything remaining
        all_tokens = []
        if self.prefilled and self.audio_embeds is not None:
            n_consumed, hit_eos, tokens = self._decode_steps(
                self.audio_embeds, self.audio_embeds.shape[0]
            )
            all_tokens.extend(tokens)
            self.audio_embeds = None

            if hit_eos:
                self.eos_text = self.full_text

        # Flush last pending token
        if self.y is not None:
            token_id = self.y.item()
            if token_id != self.eos_token_id:
                text = self.sp.decode([token_id], special_token_policy=SpecialTokenPolicy.IGNORE)
                all_tokens.append(text)
                self.full_text += text

        if self.eos_text is None:
            self.eos_text = self.full_text

        return all_tokens


def create_app(model_path: str, temperature: float = 0.0):
    """Create the FastAPI application with a loaded model."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    logger.info("Loading model: %s", model_path)
    t0 = time.monotonic()
    model, sp, config = load_model(model_path)
    logger.info("Model loaded in %.1fs", time.monotonic() - t0)

    app = FastAPI(title="voxmlx realtime server")

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket):
        await ws.accept()
        logger.info("WebSocket connected")

        session = StreamingSession(model, sp, temperature)

        await ws.send_json({"type": "session.created"})

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json(
                        {"type": "error", "message": "Invalid JSON"}
                    )
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "session.update":
                    await ws.send_json({"type": "session.updated"})

                elif msg_type == "input_audio_buffer.append":
                    audio_b64 = msg.get("audio", "")
                    if not audio_b64:
                        continue

                    pcm16_bytes = base64.b64decode(audio_b64)
                    pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
                    audio_f32 = pcm16.astype(np.float32) / 32768.0

                    tokens = session.feed_audio(audio_f32)
                    for tok in tokens:
                        await ws.send_json(
                            {
                                "type": "response.audio_transcript.delta",
                                "delta": tok,
                            }
                        )

                    if session.eos_text is not None:
                        await ws.send_json(
                            {
                                "type": "response.audio_transcript.done",
                                "text": session.eos_text,
                            }
                        )
                        session.reset()

                elif msg_type == "input_audio_buffer.commit":
                    is_final = msg.get("final", False)
                    if is_final:
                        tokens = session.finalize()
                        for tok in tokens:
                            await ws.send_json(
                                {
                                    "type": "response.audio_transcript.delta",
                                    "delta": tok,
                                }
                            )
                        await ws.send_json(
                            {
                                "type": "response.audio_transcript.done",
                                "text": session.eos_text or session.full_text,
                            }
                        )
                        session.reset()
                    # Non-final commits are no-ops (we process continuously)

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception:
            logger.exception("WebSocket error")
        finally:
            session.reset()
            mx.clear_cache()

    return app


def main():
    parser = argparse.ArgumentParser(
        description="voxmlx realtime WebSocket server"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Voxtral-Mini-4B-Realtime-6bit",
        help="Model path or HF model ID",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to listen on"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to"
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="Sampling temperature"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    import uvicorn

    app = create_app(args.model, args.temp)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
