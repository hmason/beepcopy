"""Tests for TechnoRenderer."""

from beepcopy.models import AudioSegment, Node, NodeEvent, NodeType, WaveShape
from beepcopy.renderers.techno import TechnoRenderer


class TestTechnoRenderer:
    def test_container_produces_kick(self):
        renderer = TechnoRenderer()
        node = Node(type=NodeType.DICT, depth=0, event=NodeEvent.ENTER, sibling_index=0, children_count=3)
        segments = renderer.on_node(node)
        assert len(segments) >= 1
        # Should have a low-frequency kick component
        freqs = [s.frequency for s in segments]
        assert any(f < 100 for f in freqs)

    def test_list_values_arpeggiate(self):
        renderer = TechnoRenderer()
        # Enter a list
        renderer.on_node(Node(type=NodeType.LIST, depth=0, event=NodeEvent.ENTER, sibling_index=0, children_count=3))
        # Three integer leaves
        segs = []
        for i in range(3):
            s = renderer.on_node(Node(type=NodeType.INT, depth=1, event=NodeEvent.LEAF, sibling_index=i, numeric_value=i * 100))
            segs.extend(s)
        # Each leaf should produce segments with different frequencies
        freqs = [s.frequency for s in segs]
        assert len(set(freqs)) > 1  # not all the same

    def test_numeric_value_affects_sound(self):
        renderer = TechnoRenderer()
        low = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1)
        high = Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=1000)
        low_segs = renderer.on_node(low)
        high_segs = renderer.on_node(high)
        assert low_segs[0].frequency != high_segs[0].frequency

    def test_finalize_returns_all(self):
        renderer = TechnoRenderer()
        renderer.on_node(Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=42))
        renderer.on_node(Node(type=NodeType.INT, depth=0, event=NodeEvent.LEAF, sibling_index=0, numeric_value=99))
        segments = renderer.finalize()
        assert len(segments) >= 2
