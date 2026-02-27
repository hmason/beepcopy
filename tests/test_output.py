"""Tests for beepcopy audio output (file export and playback)."""

import wave
from pathlib import Path

import numpy as np

from beepcopy.output import write_wav, get_player


class TestWriteWav:
    def test_writes_valid_wav(self, tmp_path: Path):
        buffer = np.sin(np.linspace(0, 100, 4410))  # ~0.1s of audio
        output_path = tmp_path / "test.wav"
        write_wav(buffer, str(output_path), sample_rate=44100)

        assert output_path.exists()
        with wave.open(str(output_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 44100
            assert wf.getnframes() == len(buffer)

    def test_empty_buffer(self, tmp_path: Path):
        buffer = np.array([], dtype=np.float64)
        output_path = tmp_path / "empty.wav"
        write_wav(buffer, str(output_path), sample_rate=44100)
        assert output_path.exists()


class TestGetPlayer:
    def test_returns_callable(self):
        player = get_player()
        assert callable(player)
