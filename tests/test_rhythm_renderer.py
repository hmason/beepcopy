"""Tests for RhythmRenderer."""

from beepcopy.models import AudioSegment, Node, NodeEvent, NodeType, WaveShape
from beepcopy.renderers.rhythm import RhythmRenderer


class TestRhythmRenderer:
    def test_dict_sounds_like_kick(self):
        renderer = RhythmRenderer()
        node = Node(type=NodeType.DICT, depth=0, event=NodeEvent.ENTER, sibling_index=0, children_count=2)
        segments = renderer.on_node(node)
        assert len(segments) >= 1
        # Kick = low frequency
        assert segments[0].frequency < 200

    def test_list_sounds_like_hihat(self):
        renderer = RhythmRenderer()
        node = Node(type=NodeType.LIST, depth=0, event=NodeEvent.ENTER, sibling_index=0, children_count=2)
        segments = renderer.on_node(node)
        assert len(segments) >= 1
        # Hi-hat = noise waveform
        assert segments[0].wave_shape == WaveShape.NOISE

    def test_scalar_sounds_like_snare(self):
        renderer = RhythmRenderer()
        node = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=42)
        segments = renderer.on_node(node)
        assert len(segments) >= 1

    def test_value_affects_velocity(self):
        renderer = RhythmRenderer()
        quiet = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1)
        loud = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=100)
        quiet_segs = renderer.on_node(quiet)
        loud_segs = renderer.on_node(loud)
        assert quiet_segs[0].amplitude != loud_segs[0].amplitude

    def test_finalize_returns_segments(self):
        renderer = RhythmRenderer()
        node = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1)
        renderer.on_node(node)
        segments = renderer.finalize()
        assert len(segments) >= 1
