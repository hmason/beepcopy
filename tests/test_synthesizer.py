"""Tests for beepcopy audio synthesizer."""

import numpy as np

from beepcopy.models import AudioSegment, WaveShape
from beepcopy.synthesizer import synthesize, generate_waveform


class TestGenerateWaveform:
    """Test individual waveform generation."""

    def test_sine_wave_shape(self):
        samples = generate_waveform(WaveShape.SINE, 440.0, 0.01, 44100)
        assert isinstance(samples, np.ndarray)
        assert len(samples) == int(44100 * 0.01)
        # Sine wave should be in [-1, 1]
        assert samples.max() <= 1.0
        assert samples.min() >= -1.0

    def test_square_wave_values(self):
        samples = generate_waveform(WaveShape.SQUARE, 440.0, 0.01, 44100)
        # Square wave values should be close to -1 or 1
        unique_approx = set(np.sign(samples))
        assert unique_approx <= {-1.0, 0.0, 1.0}

    def test_sawtooth_range(self):
        samples = generate_waveform(WaveShape.SAWTOOTH, 440.0, 0.01, 44100)
        assert samples.max() <= 1.0
        assert samples.min() >= -1.0

    def test_triangle_range(self):
        samples = generate_waveform(WaveShape.TRIANGLE, 440.0, 0.01, 44100)
        assert samples.max() <= 1.0
        assert samples.min() >= -1.0

    def test_noise_is_random(self):
        s1 = generate_waveform(WaveShape.NOISE, 440.0, 0.1, 44100)
        s2 = generate_waveform(WaveShape.NOISE, 440.0, 0.1, 44100)
        # Two noise generations should not be identical
        assert not np.array_equal(s1, s2)


class TestSynthesize:
    """Test full synthesis pipeline from AudioSegments to buffer."""

    def test_single_segment(self):
        segments = [
            AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SINE),
        ]
        buffer = synthesize(segments, sample_rate=44100)
        assert isinstance(buffer, np.ndarray)
        expected_length = int(44100 * 0.1)
        assert abs(len(buffer) - expected_length) <= 1

    def test_amplitude_scaling(self):
        loud = [AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SINE, amplitude=1.0)]
        quiet = [AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SINE, amplitude=0.5)]
        loud_buf = synthesize(loud, sample_rate=44100)
        quiet_buf = synthesize(quiet, sample_rate=44100)
        # Quiet buffer should have smaller peak amplitude
        assert np.abs(quiet_buf).max() < np.abs(loud_buf).max()

    def test_sequential_segments(self):
        segments = [
            AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SINE, start_time=0.0),
            AudioSegment(frequency=880.0, duration=0.1, wave_shape=WaveShape.SINE, start_time=0.1),
        ]
        buffer = synthesize(segments, sample_rate=44100)
        expected_length = int(44100 * 0.2)
        assert abs(len(buffer) - expected_length) <= 1

    def test_overlapping_segments_mix(self):
        segments = [
            AudioSegment(frequency=440.0, duration=0.2, wave_shape=WaveShape.SINE, start_time=0.0),
            AudioSegment(frequency=880.0, duration=0.2, wave_shape=WaveShape.SINE, start_time=0.0),
        ]
        buffer = synthesize(segments, sample_rate=44100)
        # Buffer should be the length of the longest segment
        expected_length = int(44100 * 0.2)
        assert abs(len(buffer) - expected_length) <= 1

    def test_empty_segments(self):
        buffer = synthesize([], sample_rate=44100)
        assert len(buffer) == 0

    def test_envelope_no_clicks(self):
        """Start and end of audio should be near zero (envelope applied)."""
        segments = [
            AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SQUARE, amplitude=1.0),
        ]
        buffer = synthesize(segments, sample_rate=44100)
        # First and last few samples should be near zero due to envelope
        assert abs(buffer[0]) < 0.05
        assert abs(buffer[-1]) < 0.05
