"""Integration tests: full pipeline from data to audio."""

from pathlib import Path

import numpy as np

from beepcopy import beepcopy, beeplisten
from beepcopy.renderers.retro import RetroRenderer
from beepcopy.renderers.rhythm import RhythmRenderer
from beepcopy.renderers.techno import TechnoRenderer


SAMPLE_DATA = {
    "users": [
        {"name": "Alice", "scores": [95, 87, 92], "active": True},
        {"name": "Bob", "scores": [78], "active": False},
    ],
    "count": 2,
    "tags": ("admin", "user"),
}


class TestFullPipeline:
    def test_retro_to_file(self, tmp_path: Path):
        path = str(tmp_path / "retro.wav")
        result = beepcopy(SAMPLE_DATA, renderer=RetroRenderer(), output=path)
        assert result == SAMPLE_DATA
        assert Path(path).stat().st_size > 1000  # non-trivial audio

    def test_rhythm_to_file(self, tmp_path: Path):
        path = str(tmp_path / "rhythm.wav")
        result = beepcopy(SAMPLE_DATA, renderer=RhythmRenderer(), output=path)
        assert result == SAMPLE_DATA
        assert Path(path).stat().st_size > 1000

    def test_techno_to_file(self, tmp_path: Path):
        path = str(tmp_path / "techno.wav")
        result = beepcopy(SAMPLE_DATA, renderer=TechnoRenderer(), output=path)
        assert result == SAMPLE_DATA
        assert Path(path).stat().st_size > 1000

    def test_beeplisten_buffer(self):
        buffer = beeplisten(SAMPLE_DATA, output="buffer")
        assert isinstance(buffer, np.ndarray)
        assert len(buffer) > 1000

    def test_all_renderers_produce_different_audio(self, tmp_path: Path):
        renderers = [RetroRenderer(), RhythmRenderer(), TechnoRenderer()]
        buffers = []
        for r in renderers:
            _, buf = beepcopy(SAMPLE_DATA, renderer=r, output="buffer")
            buffers.append(buf)
        # Each renderer should produce different audio
        assert not np.array_equal(buffers[0], buffers[1])
        assert not np.array_equal(buffers[1], buffers[2])

    def test_empty_data(self, tmp_path: Path):
        path = str(tmp_path / "empty.wav")
        result = beepcopy({}, output=path)
        assert result == {}
        assert Path(path).exists()

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": {"d": {"e": 42}}}}}
        _, buffer = beepcopy(data, output="buffer")
        assert len(buffer) > 0

    def test_large_list(self):
        data = list(range(100))
        _, buffer = beepcopy(data, output="buffer")
        assert len(buffer) > 0
