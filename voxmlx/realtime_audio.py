import base64
from dataclasses import dataclass

import numpy as np

from .audio_constants import SAMPLES_PER_TOKEN

N_LEFT_PAD_TOKENS = 32
N_RIGHT_PAD_TOKENS = 17


@dataclass(frozen=True)
class AudioChunk:
    chunk: np.ndarray | None
    pending_audio: np.ndarray
    n_real_samples: int
    first_cycle: bool


def decode_pcm16_base64(audio_b64: str | bytes) -> np.ndarray:
    """Decode little-endian PCM16 base64 into normalized float32 samples."""
    pcm16_bytes = base64.b64decode(audio_b64)
    if len(pcm16_bytes) % 2 != 0:
        raise ValueError("Invalid PCM16 payload length")

    pcm16 = np.frombuffer(pcm16_bytes, dtype="<i2")
    return pcm16.astype(np.float32) / 32768.0


def plan_stream_audio_chunk(
    pending_audio: np.ndarray,
    incoming_audio: np.ndarray | None,
    first_cycle: bool,
    samples_per_token: int = SAMPLES_PER_TOKEN,
    n_left_pad_tokens: int = N_LEFT_PAD_TOKENS,
) -> AudioChunk:
    """Append incoming audio and prepare the next full-token streaming chunk."""
    if incoming_audio is not None and incoming_audio.size:
        if pending_audio.size:
            pending_audio = np.concatenate([pending_audio, incoming_audio])
        else:
            pending_audio = incoming_audio

    if pending_audio.size < samples_per_token:
        return AudioChunk(None, pending_audio, 0, first_cycle)

    n_feed = (pending_audio.size // samples_per_token) * samples_per_token
    audio_to_feed = pending_audio[:n_feed]
    next_pending = pending_audio[n_feed:]

    if first_cycle:
        left_pad = np.zeros(n_left_pad_tokens * samples_per_token, dtype=np.float32)
        chunk = np.concatenate([left_pad, audio_to_feed])
        first_cycle = False
    else:
        chunk = audio_to_feed

    return AudioChunk(chunk, next_pending, n_feed, first_cycle)


def plan_final_audio_chunk(
    pending_audio: np.ndarray,
    first_cycle: bool,
    samples_per_token: int = SAMPLES_PER_TOKEN,
    n_left_pad_tokens: int = N_LEFT_PAD_TOKENS,
    n_right_pad_tokens: int = N_RIGHT_PAD_TOKENS,
) -> AudioChunk:
    """Prepare final audio with right padding and optional first-cycle left padding."""
    right_pad = np.zeros(n_right_pad_tokens * samples_per_token, dtype=np.float32)
    flush_chunk = np.concatenate([pending_audio, right_pad])

    pad_samples = n_right_pad_tokens * samples_per_token
    if first_cycle:
        left_pad = np.zeros(n_left_pad_tokens * samples_per_token, dtype=np.float32)
        flush_chunk = np.concatenate([left_pad, flush_chunk])
        pad_samples += n_left_pad_tokens * samples_per_token

    n_feed = (flush_chunk.size // samples_per_token) * samples_per_token
    if n_feed == 0:
        return AudioChunk(None, np.zeros(0, dtype=np.float32), 0, first_cycle)

    chunk = flush_chunk[:n_feed]
    n_real_samples = n_feed - pad_samples
    return AudioChunk(chunk, np.zeros(0, dtype=np.float32), n_real_samples, False)
