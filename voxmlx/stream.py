import argparse
import threading
import time

import mlx.core as mx
import numpy as np
import sounddevice as sd
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from . import _build_prompt_tokens, load_model
from .audio import SAMPLES_PER_TOKEN, log_mel_spectrogram_step
from .cache import RotatingKVCache

N_LEFT_PAD_TOKENS = 32
N_RIGHT_PAD_TOKENS = 17


def stream_transcribe(
    model_path: str = "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
    temperature: float = 0.0,
):
    model, sp, config = load_model(model_path)

    prompt_tokens, n_delay_tokens = _build_prompt_tokens(sp)
    prefix_len = len(prompt_tokens)
    eos_token_id = sp.eos_id

    t_cond = model.time_embedding(mx.array([n_delay_tokens], dtype=mx.float32))
    mx.eval(t_cond)

    prompt_ids = mx.array([prompt_tokens])
    text_embeds = model.language_model.embed(prompt_ids)[0]  # [prefix_len, 3072]
    mx.eval(text_embeds)

    n_layers = len(model.language_model.layers)
    sliding_window = 8192

    def sample(logits):
        if temperature <= 0:
            return mx.argmax(logits[0, -1:], axis=-1).squeeze()
        return mx.random.categorical(logits[0, -1:] / temperature).squeeze()

    def decode_steps(embeds, n_to_decode):
        """Decode n_to_decode positions from embeds[0..n_to_decode-1].

        Returns (n_consumed, hit_eos). On EOS, cache and y are reset.
        """
        nonlocal cache, y

        for i in range(n_to_decode):
            token_embed = model.language_model.embed(y.reshape(1, 1))[0, 0]
            step_embed = (embeds[i] + token_embed)[None, None, :]
            logits = model.decode(step_embed, t_cond, mask=None, cache=cache)
            next_y = sample(logits)
            mx.async_eval(next_y)

            token_id = y.item()
            if token_id == eos_token_id:
                print(flush=True)
                cache = None
                y = None
                return i, True

            text = sp.decode(
                [token_id], special_token_policy=SpecialTokenPolicy.IGNORE
            )
            print(text, end="", flush=True)

            if i > 0 and i % 256 == 0:
                mx.clear_cache()

            y = next_y

        return n_to_decode, False

    # Audio buffer and lock
    lock = threading.Lock()
    audio_buf = np.zeros(0, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        nonlocal audio_buf
        with lock:
            audio_buf = np.append(audio_buf, indata[:, 0])

    # Decoder state
    cache = None
    y = None

    # Incremental encoder state
    audio_tail = None       # mel STFT overlap (240 samples)
    conv1_tail = None       # conv1 kernel overlap (2 frames)
    conv2_tail = None       # conv2 kernel overlap (1 frame)
    encoder_cache = None    # KV cache for encoder transformer layers
    ds_buf = None           # partial downsample group

    # Bounded buffers and counters
    pending_audio = np.zeros(0, dtype=np.float32)  # unprocessed audio remainder
    audio_embeds = None     # only undecoded embeddings
    n_audio_samples_fed = 0 # total real audio samples fed (for safe decode limit)
    n_total_decoded = 0     # total positions consumed (prefill + decode)
    first_cycle = True
    prefilled = False

    def reset_all_state():
        nonlocal audio_tail, conv1_tail, conv2_tail, encoder_cache, ds_buf
        nonlocal pending_audio, audio_embeds, n_audio_samples_fed
        nonlocal n_total_decoded, first_cycle, prefilled
        audio_tail = None
        conv1_tail = None
        conv2_tail = None
        encoder_cache = None
        ds_buf = None
        pending_audio = np.zeros(0, dtype=np.float32)
        audio_embeds = None
        n_audio_samples_fed = 0
        n_total_decoded = 0
        first_cycle = True
        prefilled = False

    print("Listening... (Ctrl+C to stop)\n", flush=True)

    stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="float32",
        blocksize=SAMPLES_PER_TOKEN,
        callback=callback,
    )
    stream.start()

    try:
        start_time = time.monotonic()
        warned_no_audio = False
        while True:
            # Drain new audio from buffer (swap, not copy)
            with lock:
                new_audio = audio_buf
                audio_buf = np.zeros(0, dtype=np.float32)

            if len(new_audio) > 0:
                pending_audio = np.append(pending_audio, new_audio)

            if first_cycle and len(pending_audio) < SAMPLES_PER_TOKEN:
                elapsed = time.monotonic() - start_time
                if not warned_no_audio and elapsed > 2.0:
                    warned_no_audio = True
                    print(
                        "Warning: No audio received. Check that your terminal app "
                        "has microphone permission in System Settings > Privacy & "
                        "Security > Microphone.",
                        flush=True,
                    )
                time.sleep(0.02)
                continue

            # Encode new audio if we have at least one token's worth
            if first_cycle and len(pending_audio) >= SAMPLES_PER_TOKEN:
                # First cycle: feed left-pad + all available audio
                left_pad = np.zeros(
                    N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32
                )
                n_feed = (len(pending_audio) // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
                chunk = np.concatenate([left_pad, pending_audio[:n_feed]])
                pending_audio = pending_audio[n_feed:]
                n_audio_samples_fed += n_feed

                mel, audio_tail = log_mel_spectrogram_step(chunk, audio_tail)
                new_embeds, conv1_tail, conv2_tail, encoder_cache, ds_buf = (
                    model.encode_step(
                        mel, conv1_tail, conv2_tail, encoder_cache, ds_buf
                    )
                )
                if new_embeds is not None:
                    mx.eval(new_embeds)
                    audio_embeds = new_embeds
                first_cycle = False

            elif not first_cycle and len(pending_audio) >= SAMPLES_PER_TOKEN:
                # Subsequent cycles: feed all available audio
                n_feed = (len(pending_audio) // SAMPLES_PER_TOKEN) * SAMPLES_PER_TOKEN
                chunk = pending_audio[:n_feed]
                pending_audio = pending_audio[n_feed:]
                n_audio_samples_fed += n_feed

                mel, audio_tail = log_mel_spectrogram_step(chunk, audio_tail)
                new_embeds, conv1_tail, conv2_tail, encoder_cache, ds_buf = (
                    model.encode_step(
                        mel, conv1_tail, conv2_tail, encoder_cache, ds_buf
                    )
                )
                if new_embeds is not None:
                    mx.eval(new_embeds)
                    if audio_embeds is not None:
                        audio_embeds = mx.concatenate([audio_embeds, new_embeds])
                        mx.eval(audio_embeds)
                    else:
                        audio_embeds = new_embeds

            if audio_embeds is None:
                time.sleep(0.02)
                continue

            # How many undecoded embeddings are safe to decode
            safe_total = (
                N_LEFT_PAD_TOKENS + n_audio_samples_fed // SAMPLES_PER_TOKEN
            )
            n_decodable = min(
                audio_embeds.shape[0], safe_total - n_total_decoded
            )

            if n_decodable <= 0:
                time.sleep(0.02)
                continue

            if not prefilled:
                # Need prefix_len total positions for prefill
                if n_total_decoded + audio_embeds.shape[0] < prefix_len:
                    time.sleep(0.02)
                    continue

                cache = [RotatingKVCache(sliding_window) for _ in range(n_layers)]

                prefix_embeds = text_embeds + audio_embeds[:prefix_len]
                prefix_embeds = prefix_embeds[None, :, :]

                logits = model.decode(prefix_embeds, t_cond, "causal", cache)
                mx.eval(logits, *[x for c in cache for x in (c.keys, c.values)])

                y = sample(logits)
                mx.async_eval(y)

                # Trim consumed prefix
                audio_embeds = audio_embeds[prefix_len:]
                n_total_decoded = prefix_len
                prefilled = True

                # Recompute decodable after consuming prefix
                n_decodable = min(
                    audio_embeds.shape[0], safe_total - n_total_decoded
                )

            if n_decodable <= 0:
                time.sleep(0.02)
                continue

            # Decode new positions
            n_consumed, hit_eos = decode_steps(audio_embeds, n_decodable)
            n_total_decoded += n_consumed

            # Trim consumed embeddings
            if audio_embeds.shape[0] > n_consumed:
                audio_embeds = audio_embeds[n_consumed:]
            else:
                audio_embeds = None

            if hit_eos:
                reset_all_state()

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()

        # Final flush: feed remaining audio + right padding through incremental
        # pipeline, then decode all remaining positions.
        if cache is not None and y is not None:
            with lock:
                final_audio = audio_buf
                audio_buf = np.zeros(0, dtype=np.float32)

            pending_audio = np.append(pending_audio, final_audio)
            right_pad = np.zeros(
                N_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32
            )
            flush_chunk = np.concatenate([pending_audio, right_pad])
            mel, audio_tail = log_mel_spectrogram_step(flush_chunk, audio_tail)
            new_embeds, conv1_tail, conv2_tail, encoder_cache, ds_buf = (
                model.encode_step(
                    mel, conv1_tail, conv2_tail, encoder_cache, ds_buf
                )
            )
            if new_embeds is not None:
                mx.eval(new_embeds)
                if audio_embeds is not None:
                    audio_embeds = mx.concatenate([audio_embeds, new_embeds])
                    mx.eval(audio_embeds)
                else:
                    audio_embeds = new_embeds
            if audio_embeds is not None:
                decode_steps(audio_embeds, audio_embeds.shape[0])

        # Flush last pending token
        if y is not None:
            token_id = y.item()
            if token_id != eos_token_id:
                text = sp.decode(
                    [token_id], special_token_policy=SpecialTokenPolicy.IGNORE
                )
                print(text, end="", flush=True)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Live streaming speech-to-text with Voxtral"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Voxtral-Mini-4B-Realtime-6bit",
        help="Model path or HF model ID",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    args = parser.parse_args()

    stream_transcribe(
        model_path=args.model,
        temperature=args.temp,
    )
