"""Tests for beepcopy data models."""

from beepcopy.models import Node, AudioSegment, NodeEvent, NodeType, WaveShape


class TestNode:
    def test_leaf_node(self):
        node = Node(
            type=NodeType.INT,
            depth=0,
            event=NodeEvent.LEAF,
            sibling_index=0,
            numeric_value=42,
        )
        assert node.type == NodeType.INT
        assert node.depth == 0
        assert node.event == NodeEvent.LEAF
        assert node.numeric_value == 42
        assert node.children_count is None
        assert node.key is None

    def test_container_enter_node(self):
        node = Node(
            type=NodeType.DICT,
            depth=0,
            event=NodeEvent.ENTER,
            sibling_index=0,
            children_count=3,
        )
        assert node.type == NodeType.DICT
        assert node.event == NodeEvent.ENTER
        assert node.children_count == 3

    def test_node_with_key(self):
        node = Node(
            type=NodeType.STR,
            depth=1,
            event=NodeEvent.LEAF,
            sibling_index=0,
            string_length=5,
            string_hash=12345,
            key="name",
        )
        assert node.key == "name"
        assert node.string_length == 5
        assert node.string_hash == 12345


class TestAudioSegment:
    def test_audio_segment(self):
        seg = AudioSegment(
            frequency=440.0,
            duration=0.1,
            wave_shape=WaveShape.SINE,
            amplitude=0.8,
            start_time=0.0,
        )
        assert seg.frequency == 440.0
        assert seg.duration == 0.1
        assert seg.wave_shape == WaveShape.SINE
        assert seg.amplitude == 0.8
        assert seg.start_time == 0.0

    def test_audio_segment_defaults(self):
        seg = AudioSegment(
            frequency=440.0,
            duration=0.1,
            wave_shape=WaveShape.SINE,
        )
        assert seg.amplitude == 1.0
        assert seg.start_time == 0.0
