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
    scratch_buffer: np.ndarray | None = None


def decode_pcm16_base64(audio_b64: str | bytes) -> np.ndarray:
    """Decode little-endian PCM16 base64 into normalized float32 samples."""
    pcm16_bytes = base64.b64decode(audio_b64)
    if len(pcm16_bytes) % 2 != 0:
        raise ValueError("Invalid PCM16 payload length")

    pcm16 = np.frombuffer(pcm16_bytes, dtype="<i2")
    return pcm16.astype(np.float32) / 32768.0


def _empty_float32(size: int, scratch_buffer: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if (
        scratch_buffer is not None
        and scratch_buffer.dtype == np.float32
        and scratch_buffer.size >= size
    ):
        return scratch_buffer[:size], scratch_buffer

    buffer = np.empty(size, dtype=np.float32)
    return buffer, buffer


def _combine_pending(
    pending_audio: np.ndarray,
    incoming_audio: np.ndarray | None,
    total_size: int,
) -> np.ndarray:
    if pending_audio.size == 0:
        return incoming_audio if incoming_audio is not None else pending_audio
    if incoming_audio is None or incoming_audio.size == 0:
        return pending_audio

    combined = np.empty(total_size, dtype=np.float32)
    combined[: pending_audio.size] = pending_audio
    combined[pending_audio.size :] = incoming_audio
    return combined


def _remaining_audio(
    pending_audio: np.ndarray,
    incoming_audio: np.ndarray | None,
    consumed: int,
    total_size: int,
) -> np.ndarray:
    remaining_size = total_size - consumed
    if remaining_size == 0:
        return np.empty(0, dtype=np.float32)

    if consumed < pending_audio.size:
        pending_remainder = pending_audio[consumed:]
        if incoming_audio is None or incoming_audio.size == 0:
            return pending_remainder
        return _combine_pending(
            pending_remainder,
            incoming_audio,
            pending_remainder.size + incoming_audio.size,
        )

    if incoming_audio is None or incoming_audio.size == 0:
        return np.empty(0, dtype=np.float32)
    return incoming_audio[consumed - pending_audio.size :]


def _copy_audio_prefix(
    out: np.ndarray,
    pending_audio: np.ndarray,
    incoming_audio: np.ndarray | None,
    count: int,
) -> None:
    write_pos = 0
    if pending_audio.size:
        n_pending = min(pending_audio.size, count)
        out[:n_pending] = pending_audio[:n_pending]
        write_pos = n_pending

    remaining = count - write_pos
    if remaining > 0:
        if incoming_audio is None:
            raise ValueError("Not enough audio to fill output buffer")
        out[write_pos:count] = incoming_audio[:remaining]


def plan_stream_audio_chunk(
    pending_audio: np.ndarray,
    incoming_audio: np.ndarray | None,
    first_cycle: bool,
    samples_per_token: int = SAMPLES_PER_TOKEN,
    n_left_pad_tokens: int = N_LEFT_PAD_TOKENS,
    scratch_buffer: np.ndarray | None = None,
) -> AudioChunk:
    """Append incoming audio and prepare the next full-token streaming chunk."""
    incoming_size = 0 if incoming_audio is None else incoming_audio.size
    total_size = pending_audio.size + incoming_size

    if total_size < samples_per_token:
        pending = _combine_pending(pending_audio, incoming_audio, total_size)
        return AudioChunk(None, pending, 0, first_cycle, scratch_buffer)

    n_feed = (total_size // samples_per_token) * samples_per_token
    next_pending = _remaining_audio(
        pending_audio, incoming_audio, n_feed, total_size
    )

    if first_cycle:
        left_pad = n_left_pad_tokens * samples_per_token
        chunk, scratch_buffer = _empty_float32(left_pad + n_feed, scratch_buffer)
        chunk[:left_pad] = 0.0
        _copy_audio_prefix(
            chunk[left_pad:],
            pending_audio,
            incoming_audio,
            n_feed,
        )
        first_cycle = False
    elif pending_audio.size == 0 and incoming_audio is not None:
        chunk = incoming_audio[:n_feed]
    elif n_feed <= pending_audio.size:
        chunk = pending_audio[:n_feed]
    else:
        chunk, scratch_buffer = _empty_float32(n_feed, scratch_buffer)
        _copy_audio_prefix(chunk, pending_audio, incoming_audio, n_feed)

    return AudioChunk(chunk, next_pending, n_feed, first_cycle, scratch_buffer)


def plan_final_audio_chunk(
    pending_audio: np.ndarray,
    first_cycle: bool,
    samples_per_token: int = SAMPLES_PER_TOKEN,
    n_left_pad_tokens: int = N_LEFT_PAD_TOKENS,
    n_right_pad_tokens: int = N_RIGHT_PAD_TOKENS,
    scratch_buffer: np.ndarray | None = None,
) -> AudioChunk:
    """Prepare final audio with right padding and optional first-cycle left padding."""
    pad_samples = n_right_pad_tokens * samples_per_token
    total_size = pending_audio.size + pad_samples
    if first_cycle:
        left_pad = n_left_pad_tokens * samples_per_token
        total_size += left_pad
        pad_samples += left_pad
    else:
        left_pad = 0

    n_feed = (total_size // samples_per_token) * samples_per_token
    if n_feed == 0:
        return AudioChunk(
            None,
            np.empty(0, dtype=np.float32),
            0,
            first_cycle,
            scratch_buffer,
        )

    chunk, scratch_buffer = _empty_float32(n_feed, scratch_buffer)
    write_pos = 0
    remaining = n_feed

    if left_pad:
        n = min(left_pad, remaining)
        chunk[:n] = 0.0
        write_pos = n
        remaining -= n

    if remaining > 0:
        n = min(pending_audio.size, remaining)
        if n:
            chunk[write_pos : write_pos + n] = pending_audio[:n]
            write_pos += n
            remaining -= n

    if remaining > 0:
        chunk[write_pos : write_pos + remaining] = 0.0

    n_real_samples = n_feed - pad_samples
    return AudioChunk(
        chunk,
        np.empty(0, dtype=np.float32),
        n_real_samples,
        False,
        scratch_buffer,
    )
