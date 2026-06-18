"""
WebSocket server for voxmlx that speaks the OpenAI Realtime API protocol.

Start with:  voxmlx-serve [--model MODEL] [--port PORT] [--temp TEMP]
Or:          python -m voxmlx.server
"""

import argparse
import json
import logging
import time

import mlx.core as mx
import numpy as np
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from . import _build_prompt_tokens, load_model
from .audio import log_mel_spectrogram_step
from .audio_constants import SAMPLES_PER_TOKEN
from .cache import RotatingKVCache
from .realtime_audio import (
    N_LEFT_PAD_TOKENS,
    decode_pcm16_base64,
    plan_final_audio_chunk,
    plan_stream_audio_chunk,
)

logger = logging.getLogger("voxmlx.server")


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

            self.y = next_y

        return n_to_decode, False, tokens

    def _encode_audio(self, incoming_audio: np.ndarray | None = None):
        """Encode pending audio into embeddings. Returns True if new embeds produced."""
        planned = plan_stream_audio_chunk(
            self.pending_audio, incoming_audio, self.first_cycle
        )
        if planned.chunk is None:
            return False

        self.pending_audio = planned.pending_audio
        self.n_audio_samples_fed += planned.n_real_samples
        self.first_cycle = planned.first_cycle

        mel, self.audio_tail = log_mel_spectrogram_step(
            planned.chunk, self.audio_tail
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
                mx.eval(self.audio_embeds)
            else:
                self.audio_embeds = new_embeds
        return True

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
        self.eos_text = None

        all_tokens = []
        incoming_audio = audio_f32

        # Encode all available audio
        while True:
            encoded = self._encode_audio(incoming_audio)
            incoming_audio = None
            if not encoded:
                break

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
        planned = plan_final_audio_chunk(self.pending_audio, was_first_cycle)
        self.pending_audio = np.zeros(0, dtype=np.float32)
        self.first_cycle = planned.first_cycle
        if planned.chunk is None:
            return []
        self.n_audio_samples_fed += planned.n_real_samples

        mel, self.audio_tail = log_mel_spectrogram_step(planned.chunk, self.audio_tail)
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
                mx.eval(self.audio_embeds)
            else:
                self.audio_embeds = new_embeds

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
        ws_start = time.monotonic()
        last_stats_log = ws_start
        received_samples = 0
        append_count = 0

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
                    session_cfg = msg.get("session") if isinstance(msg, dict) else None
                    if isinstance(session_cfg, dict):
                        input_fmt = session_cfg.get("input_audio_format")
                        if input_fmt is not None:
                            logger.info(
                                "session.update input_audio_format=%s", input_fmt
                            )
                    await ws.send_json({"type": "session.updated"})

                elif msg_type == "input_audio_buffer.append":
                    audio_b64 = msg.get("audio", "")
                    if not audio_b64:
                        continue

                    try:
                        audio_f32 = decode_pcm16_base64(audio_b64)
                    except ValueError:
                        await ws.send_json(
                            {
                                "type": "error",
                                "message": "Invalid PCM16 payload length",
                            }
                        )
                        continue
                    received_samples += audio_f32.shape[0]
                    append_count += 1

                    now = time.monotonic()
                    if now - last_stats_log >= 5.0:
                        elapsed = now - ws_start
                        est_sr = received_samples / elapsed if elapsed > 0 else 0.0
                        rms = (
                            float(np.sqrt(np.mean(audio_f32.astype(np.float64) ** 2)))
                            if audio_f32.size
                            else 0.0
                        )
                        peak = (
                            float(np.max(np.abs(audio_f32)))
                            if audio_f32.size
                            else 0.0
                        )
                        dec_offset = (
                            session.cache[0].offset
                            if session.cache is not None and len(session.cache) > 0
                            else 0
                        )
                        enc_offset = (
                            session.encoder_cache[0].offset
                            if session.encoder_cache is not None
                            and len(session.encoder_cache) > 0
                            else 0
                        )
                        embed_backlog = (
                            int(session.audio_embeds.shape[0])
                            if session.audio_embeds is not None
                            else 0
                        )
                        logger.info(
                            "audio ingest appends=%d samples=%d est_sr=%.1fHz chunk=%d rms=%.4f peak=%.4f dec_off=%d enc_off=%d decoded=%d embeds=%d pend=%d",
                            append_count,
                            received_samples,
                            est_sr,
                            audio_f32.size,
                            rms,
                            peak,
                            dec_offset,
                            enc_offset,
                            session.n_total_decoded,
                            embed_backlog,
                            len(session.pending_audio),
                        )
                        last_stats_log = now

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
                    # Only finalize/reset when the client explicitly marks
                    # commit as final; non-final commits remain no-ops.
                    is_final = msg.get("final", False)
                    logger.info(
                        "input_audio_buffer.commit final=%s pending_samples=%d prefilled=%s",
                        is_final,
                        len(session.pending_audio),
                        session.prefilled,
                    )
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

                elif msg_type == "input_audio_buffer.clear":
                    session.reset()
                    await ws.send_json({"type": "input_audio_buffer.cleared"})

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
