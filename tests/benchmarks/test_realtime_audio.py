import base64

import numpy as np
import pytest

from voxmlx.audio_constants import SAMPLES_PER_TOKEN
from voxmlx.realtime_audio import (
    N_LEFT_PAD_TOKENS,
    N_RIGHT_PAD_TOKENS,
    decode_pcm16_base64,
    plan_final_audio_chunk,
    plan_stream_audio_chunk,
)


def _pcm16_samples(n_samples: int) -> np.ndarray:
    values = (np.arange(n_samples, dtype=np.int32) * 97) % 65536
    return (values - 32768).astype("<i2")


def _audio_samples(n_samples: int) -> np.ndarray:
    values = (np.arange(n_samples, dtype=np.float32) % 400.0) - 200.0
    return values / 200.0


def test_decode_pcm16_base64(benchmark):
    pcm16 = _pcm16_samples(16_000)
    audio_b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")

    decoded = benchmark(decode_pcm16_base64, audio_b64)

    assert decoded.dtype == np.float32
    assert decoded.shape == (16_000,)
    np.testing.assert_allclose(decoded[:16], pcm16[:16].astype(np.float32) / 32768.0)


def test_decode_pcm16_base64_rejects_odd_payload():
    audio_b64 = base64.b64encode(b"\x00").decode("ascii")

    with pytest.raises(ValueError, match="Invalid PCM16 payload length"):
        decode_pcm16_base64(audio_b64)


def test_plan_stream_audio_chunk_first_cycle(benchmark):
    pending_audio = _audio_samples(SAMPLES_PER_TOKEN // 2)
    incoming_audio = _audio_samples(4 * SAMPLES_PER_TOKEN + 123)
    combined = np.concatenate([pending_audio, incoming_audio])
    expected_feed = 4 * SAMPLES_PER_TOKEN

    planned = benchmark(
        plan_stream_audio_chunk,
        pending_audio,
        incoming_audio,
        True,
    )

    assert planned.n_real_samples == expected_feed
    assert planned.first_cycle is False
    assert planned.pending_audio.shape == (combined.size - expected_feed,)
    assert planned.chunk is not None
    assert planned.chunk.shape == (
        (N_LEFT_PAD_TOKENS + 4) * SAMPLES_PER_TOKEN,
    )
    np.testing.assert_array_equal(
        planned.chunk[: N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN],
        np.zeros(N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        planned.chunk[N_LEFT_PAD_TOKENS * SAMPLES_PER_TOKEN :],
        combined[:expected_feed],
    )


def test_plan_stream_audio_chunk_steady_state_view(benchmark):
    pending_audio = np.empty(0, dtype=np.float32)
    incoming_audio = _audio_samples(4 * SAMPLES_PER_TOKEN + 123)
    expected_feed = 4 * SAMPLES_PER_TOKEN

    planned = benchmark(
        plan_stream_audio_chunk,
        pending_audio,
        incoming_audio,
        False,
    )

    assert planned.n_real_samples == expected_feed
    assert planned.first_cycle is False
    assert planned.chunk is not None
    assert planned.chunk.shape == (expected_feed,)
    assert np.shares_memory(planned.chunk, incoming_audio)
    assert np.shares_memory(planned.pending_audio, incoming_audio)
    assert planned.scratch_buffer is None
    np.testing.assert_array_equal(planned.chunk, incoming_audio[:expected_feed])


def test_plan_stream_audio_chunk_split_reuses_scratch(benchmark):
    pending_audio = _audio_samples(SAMPLES_PER_TOKEN // 2)
    incoming_audio = _audio_samples(4 * SAMPLES_PER_TOKEN + 123)
    scratch = np.empty(4 * SAMPLES_PER_TOKEN, dtype=np.float32)
    combined = np.concatenate([pending_audio, incoming_audio])
    expected_feed = 4 * SAMPLES_PER_TOKEN

    planned = benchmark(
        plan_stream_audio_chunk,
        pending_audio,
        incoming_audio,
        False,
        SAMPLES_PER_TOKEN,
        N_LEFT_PAD_TOKENS,
        scratch,
    )

    assert planned.n_real_samples == expected_feed
    assert planned.first_cycle is False
    assert planned.chunk is not None
    assert planned.chunk.shape == (expected_feed,)
    assert planned.scratch_buffer is scratch
    assert np.shares_memory(planned.chunk, scratch)
    assert np.shares_memory(planned.pending_audio, incoming_audio)
    np.testing.assert_array_equal(planned.chunk, combined[:expected_feed])


def test_plan_final_audio_chunk(benchmark):
    pending_audio = _audio_samples(2 * SAMPLES_PER_TOKEN + 320)
    right_pad_samples = N_RIGHT_PAD_TOKENS * SAMPLES_PER_TOKEN
    expected_feed = 2 * SAMPLES_PER_TOKEN
    scratch = np.empty(expected_feed + right_pad_samples, dtype=np.float32)

    planned = benchmark(
        plan_final_audio_chunk,
        pending_audio,
        False,
        SAMPLES_PER_TOKEN,
        N_LEFT_PAD_TOKENS,
        N_RIGHT_PAD_TOKENS,
        scratch,
    )

    assert planned.n_real_samples == expected_feed
    assert planned.first_cycle is False
    assert planned.pending_audio.shape == (0,)
    assert planned.chunk is not None
    assert planned.chunk.shape == (expected_feed + right_pad_samples,)
    assert planned.scratch_buffer is scratch
    assert np.shares_memory(planned.chunk, scratch)
    np.testing.assert_array_equal(planned.chunk[: pending_audio.size], pending_audio)
    np.testing.assert_array_equal(
        planned.chunk[pending_audio.size :],
        np.zeros(
            expected_feed + right_pad_samples - pending_audio.size,
            dtype=np.float32,
        ),
    )
