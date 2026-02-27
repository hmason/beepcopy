"""Audio output: WAV file export and playback."""

from __future__ import annotations

import struct
import tempfile
import threading
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from beepcopy.synthesizer import DEFAULT_SAMPLE_RATE


def write_wav(
    buffer: np.ndarray,
    path: str,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """Write an audio buffer to a WAV file (16-bit mono)."""
    # Convert float64 [-1, 1] to int16
    if len(buffer) == 0:
        int_data = b""
    else:
        clipped = np.clip(buffer, -1.0, 1.0)
        int_data = (clipped * 32767).astype(np.int16).tobytes()

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_data)


def _play_simpleaudio(buffer: np.ndarray, sample_rate: int) -> None:
    """Play audio using simpleaudio."""
    import simpleaudio as sa  # type: ignore[import-untyped]

    int_data = (np.clip(buffer, -1.0, 1.0) * 32767).astype(np.int16)
    play_obj = sa.play_buffer(int_data, 1, 2, sample_rate)
    play_obj.wait_done()


def _play_system(buffer: np.ndarray, sample_rate: int) -> None:
    """Fallback: write to temp WAV and open with system player."""
    import subprocess
    import sys

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    write_wav(buffer, tmp_path, sample_rate=sample_rate)

    if sys.platform == "darwin":
        subprocess.Popen(["afplay", tmp_path])
    elif sys.platform.startswith("linux"):
        subprocess.Popen(["aplay", tmp_path])
    elif sys.platform == "win32":
        import winsound

        winsound.PlaySound(tmp_path, winsound.SND_FILENAME)


def get_player() -> Callable[[np.ndarray, int], None]:
    """Return the best available playback function."""
    try:
        import simpleaudio  # type: ignore[import-untyped]  # noqa: F401

        return _play_simpleaudio
    except ImportError:
        return _play_system


def play(
    buffer: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    blocking: bool = False,
) -> None:
    """Play an audio buffer through speakers."""
    if len(buffer) == 0:
        return

    player = get_player()
    if blocking:
        player(buffer, sample_rate)
    else:
        thread = threading.Thread(
            target=player, args=(buffer, sample_rate), daemon=True
        )
        thread.start()
