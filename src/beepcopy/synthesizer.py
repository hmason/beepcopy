"""Audio synthesis from AudioSegment descriptors using numpy."""

from __future__ import annotations

import numpy as np

from beepcopy.models import AudioSegment, WaveShape

DEFAULT_SAMPLE_RATE = 44100


def generate_waveform(
    shape: WaveShape,
    frequency: float,
    duration: float,
    sample_rate: int,
) -> np.ndarray:
    """Generate a raw waveform of the given shape."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    match shape:
        case WaveShape.SINE:
            return np.sin(2 * np.pi * frequency * t)
        case WaveShape.SQUARE:
            return np.sign(np.sin(2 * np.pi * frequency * t))
        case WaveShape.SAWTOOTH:
            return 2 * (frequency * t % 1) - 1
        case WaveShape.TRIANGLE:
            return 2 * np.abs(2 * (frequency * t % 1) - 1) - 1
        case WaveShape.NOISE:
            rng = np.random.default_rng()
            return rng.uniform(-1.0, 1.0, num_samples)
        case _:
            raise ValueError(f"Unsupported wave shape: {shape}")


def _apply_envelope(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply an attack/decay envelope to prevent clicks."""
    n = len(samples)
    if n == 0:
        return samples

    # 5ms attack and decay
    fade_samples = min(int(sample_rate * 0.005), n // 2)
    if fade_samples == 0:
        return samples

    envelope = np.ones(n)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return samples * envelope


def synthesize(
    segments: list[AudioSegment],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a list of AudioSegments into a single audio buffer."""
    if not segments:
        return np.array([], dtype=np.float64)

    # Determine total buffer length
    end_times = [seg.start_time + seg.duration for seg in segments]
    total_duration = max(end_times)
    total_samples = int(sample_rate * total_duration)
    buffer = np.zeros(total_samples, dtype=np.float64)

    for seg in segments:
        waveform = generate_waveform(seg.wave_shape, seg.frequency, seg.duration, sample_rate)
        waveform = _apply_envelope(waveform, sample_rate)
        waveform *= seg.amplitude

        start_sample = int(seg.start_time * sample_rate)
        end_sample = start_sample + len(waveform)
        # Trim if it overflows
        if end_sample > total_samples:
            waveform = waveform[:total_samples - start_sample]
            end_sample = total_samples
        buffer[start_sample:end_sample] += waveform

    # Normalize to prevent clipping
    peak = np.abs(buffer).max()
    if peak > 1.0:
        buffer /= peak

    return buffer
