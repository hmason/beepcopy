"""Tests for beepcopy renderers."""

from beepcopy.models import AudioSegment, Node, NodeEvent, NodeType, WaveShape
from beepcopy.renderers import BaseRenderer
from beepcopy.renderers.retro import RetroRenderer


class TestBaseRenderer:
    def test_on_node_returns_list(self):
        renderer = BaseRenderer()
        node = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0)
        result = renderer.on_node(node)
        assert isinstance(result, list)

    def test_finalize_returns_list(self):
        renderer = BaseRenderer()
        result = renderer.finalize()
        assert isinstance(result, list)


class TestRetroRenderer:
    def test_leaf_int_produces_segment(self):
        renderer = RetroRenderer()
        node = Node(
            type=NodeType.INT, depth=0, event=NodeEvent.LEAF,
            sibling_index=0, numeric_value=42,
        )
        segments = renderer.on_node(node)
        assert len(segments) >= 1
        assert all(isinstance(s, AudioSegment) for s in segments)

    def test_deeper_nodes_lower_pitch(self):
        renderer = RetroRenderer()
        shallow = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1)
        deep = Node(type=NodeType.INT, depth=3, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1)
        shallow_segs = renderer.on_node(shallow)
        deep_segs = renderer.on_node(deep)
        assert shallow_segs[0].frequency > deep_segs[0].frequency

    def test_different_types_different_waveforms(self):
        renderer = RetroRenderer()
        int_node = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1)
        str_node = Node(type=NodeType.STR, depth=0, event=NodeEvent.LEAF, sibling_index=0, string_length=3, string_hash=123)
        int_segs = renderer.on_node(int_node)
        str_segs = renderer.on_node(str_node)
        assert int_segs[0].wave_shape != str_segs[0].wave_shape

    def test_enter_event_produces_segment(self):
        renderer = RetroRenderer()
        node = Node(type=NodeType.LIST, depth=0, event=NodeEvent.ENTER, sibling_index=0, children_count=3)
        segments = renderer.on_node(node)
        assert len(segments) >= 1

    def test_exit_event_produces_segment(self):
        renderer = RetroRenderer()
        node = Node(type=NodeType.LIST, depth=0, event=NodeEvent.EXIT, sibling_index=0)
        segments = renderer.on_node(node)
        assert len(segments) >= 1

    def test_numeric_value_modulates_frequency(self):
        renderer = RetroRenderer()
        low = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1)
        high = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1000)
        low_segs = renderer.on_node(low)
        high_segs = renderer.on_node(high)
        assert low_segs[0].frequency != high_segs[0].frequency

    def test_finalize_returns_accumulated_segments(self):
        renderer = RetroRenderer()
        node = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=42)
        renderer.on_node(node)
        renderer.on_node(node)
        segments = renderer.finalize()
        assert len(segments) >= 2

    def test_tempo_affects_duration(self):
        slow = RetroRenderer(tempo=60)
        fast = RetroRenderer(tempo=240)
        node = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1)
        slow_segs = slow.on_node(node)
        fast_segs = fast.on_node(node)
        assert slow_segs[0].duration > fast_segs[0].duration
