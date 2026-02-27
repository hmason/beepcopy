"""Tests for beepcopy data structure traverser."""

from beepcopy.traverser import traverse
from beepcopy.models import NodeEvent, NodeType


class TestTraverseScalars:
    def test_int(self):
        nodes = list(traverse(42))
        assert len(nodes) == 1
        assert nodes[0].type == NodeType.INT
        assert nodes[0].event == NodeEvent.LEAF
        assert nodes[0].depth == 0
        assert nodes[0].numeric_value == 42

    def test_float(self):
        nodes = list(traverse(3.14))
        assert len(nodes) == 1
        assert nodes[0].type == NodeType.FLOAT
        assert nodes[0].numeric_value == 3.14

    def test_string(self):
        nodes = list(traverse("hello"))
        assert len(nodes) == 1
        assert nodes[0].type == NodeType.STR
        assert nodes[0].string_length == 5
        assert nodes[0].string_hash is not None

    def test_bool(self):
        nodes = list(traverse(True))
        assert len(nodes) == 1
        assert nodes[0].type == NodeType.BOOL
        assert nodes[0].bool_value is True

    def test_none(self):
        nodes = list(traverse(None))
        assert len(nodes) == 1
        assert nodes[0].type == NodeType.NONE

    def test_string_hash_deterministic(self):
        hash1 = list(traverse("test"))[0].string_hash
        hash2 = list(traverse("test"))[0].string_hash
        assert hash1 == hash2

    def test_different_strings_different_hash(self):
        hash1 = list(traverse("alpha"))[0].string_hash
        hash2 = list(traverse("beta"))[0].string_hash
        assert hash1 != hash2


class TestTraverseContainers:
    def test_empty_list(self):
        nodes = list(traverse([]))
        assert len(nodes) == 2  # enter + exit
        assert nodes[0].event == NodeEvent.ENTER
        assert nodes[0].type == NodeType.LIST
        assert nodes[0].children_count == 0
        assert nodes[0].emptiness is True
        assert nodes[1].event == NodeEvent.EXIT

    def test_list_of_ints(self):
        nodes = list(traverse([10, 20, 30]))
        assert len(nodes) == 5  # enter + 3 leaves + exit
        assert nodes[0].event == NodeEvent.ENTER
        assert nodes[0].children_count == 3
        # Check sibling indices
        assert nodes[1].sibling_index == 0
        assert nodes[2].sibling_index == 1
        assert nodes[3].sibling_index == 2
        assert nodes[4].event == NodeEvent.EXIT

    def test_dict(self):
        nodes = list(traverse({"a": 1}))
        # enter dict, leaf str (key), leaf int (value), exit dict
        assert nodes[0].event == NodeEvent.ENTER
        assert nodes[0].type == NodeType.DICT
        assert nodes[0].children_count == 1
        # Key node
        assert nodes[1].type == NodeType.STR
        assert nodes[1].string_length == 1
        # Value node has key set
        assert nodes[2].type == NodeType.INT
        assert nodes[2].key == "a"
        assert nodes[2].numeric_value == 1
        assert nodes[3].event == NodeEvent.EXIT

    def test_tuple(self):
        nodes = list(traverse((1, 2)))
        assert nodes[0].type == NodeType.TUPLE
        assert nodes[0].children_count == 2

    def test_set(self):
        nodes = list(traverse({1}))
        assert nodes[0].type == NodeType.SET
        assert nodes[0].children_count == 1


class TestTraverseNested:
    def test_nested_dict_list(self):
        """Test {"a": [1, 2]} from the design doc."""
        data = {"a": [1, 2]}
        nodes = list(traverse(data))
        events = [(n.event, n.type, n.depth) for n in nodes]
        assert events == [
            (NodeEvent.ENTER, NodeType.DICT, 0),
            (NodeEvent.LEAF, NodeType.STR, 1),      # key "a"
            (NodeEvent.ENTER, NodeType.LIST, 1),     # value [1, 2]
            (NodeEvent.LEAF, NodeType.INT, 2),       # 1
            (NodeEvent.LEAF, NodeType.INT, 2),       # 2
            (NodeEvent.EXIT, NodeType.LIST, 1),
            (NodeEvent.EXIT, NodeType.DICT, 0),
        ]
        # The list node should have key="a"
        list_enter = nodes[2]
        assert list_enter.key == "a"

    def test_depth_tracking(self):
        data = [[["deep"]]]
        nodes = list(traverse(data))
        depths = [n.depth for n in nodes]
        assert depths == [0, 1, 2, 3, 2, 1, 0]  # enter/leaf/exit at varying depths

    def test_other_type(self):
        """Unknown types get NodeType.OTHER."""
        class Custom:
            pass
        nodes = list(traverse(Custom()))
        assert nodes[0].type == NodeType.OTHER
