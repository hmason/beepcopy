"""Tests for beepcopy public API."""

from pathlib import Path

import numpy as np

from beepcopy import beepcopy, beeplisten
from beepcopy.renderers.retro import RetroRenderer


class TestBeepcopy:
    def test_returns_deep_copy(self):
        data = {"a": [1, 2, 3]}
        result = beepcopy(data, silent=True)
        assert result == data
        assert result is not data
        assert result["a"] is not data["a"]

    def test_silent_mode(self):
        """Silent mode should copy without producing audio."""
        data = [1, 2, 3]
        result = beepcopy(data, silent=True)
        assert result == data

    def test_output_to_file(self, tmp_path: Path):
        data = {"x": [1, 2]}
        output_path = str(tmp_path / "output.wav")
        result = beepcopy(data, output=output_path)
        assert result == data
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_output_buffer(self):
        data = [1, 2, 3]
        result = beepcopy(data, output="buffer")
        # When output="buffer", returns (copy, buffer) tuple
        assert isinstance(result, tuple)
        copy, buffer = result
        assert copy == data
        assert isinstance(buffer, np.ndarray)
        assert len(buffer) > 0

    def test_custom_renderer(self, tmp_path: Path):
        data = [1, 2]
        renderer = RetroRenderer(tempo=240)
        output_path = str(tmp_path / "custom.wav")
        result = beepcopy(data, renderer=renderer, output=output_path)
        assert result == data
        assert Path(output_path).exists()

    def test_nested_structure(self):
        data = {"users": [{"name": "Alice", "scores": [95, 87]}, {"name": "Bob"}]}
        result = beepcopy(data, silent=True)
        assert result == data
        assert result is not data


class TestBeeplisten:
    def test_returns_none(self):
        result = beeplisten([1, 2, 3], silent=True)
        assert result is None

    def test_output_to_file(self, tmp_path: Path):
        output_path = str(tmp_path / "listen.wav")
        beeplisten({"a": 1}, output=output_path)
        assert Path(output_path).exists()

    def test_output_buffer(self):
        result = beeplisten([1, 2], output="buffer")
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
