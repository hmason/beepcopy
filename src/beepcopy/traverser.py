"""Depth-first traversal of Python data structures into Node streams."""

from __future__ import annotations

from collections.abc import Generator
from hashlib import md5

from beepcopy.models import Node, NodeEvent, NodeType

# Map Python types to NodeType. Order matters: bool before int (bool is subclass of int).
_TYPE_MAP: list[tuple[type, NodeType]] = [
    (bool, NodeType.BOOL),
    (int, NodeType.INT),
    (float, NodeType.FLOAT),
    (str, NodeType.STR),
    (dict, NodeType.DICT),
    (list, NodeType.LIST),
    (tuple, NodeType.TUPLE),
    (set, NodeType.SET),
]

_CONTAINER_TYPES = {NodeType.DICT, NodeType.LIST, NodeType.TUPLE, NodeType.SET}


def _classify(value: object) -> NodeType:
    """Determine the NodeType for a Python value."""
    if value is None:
        return NodeType.NONE
    for py_type, node_type in _TYPE_MAP:
        if isinstance(value, py_type):
            return node_type
    return NodeType.OTHER


def _string_hash(s: str) -> int:
    """Deterministic hash of a string for consistent sonification."""
    return int(md5(s.encode("utf-8")).hexdigest()[:8], 16)


def traverse(
    value: object,
    *,
    _depth: int = 0,
    _sibling_index: int = 0,
    _key: str | None = None,
) -> Generator[Node]:
    """Walk a data structure depth-first, yielding Node objects."""
    node_type = _classify(value)

    if node_type in _CONTAINER_TYPES:
        yield from _traverse_container(value, node_type, _depth, _sibling_index, _key)
    else:
        yield _make_leaf(value, node_type, _depth, _sibling_index, _key)


def _traverse_container(
    value: object,
    node_type: NodeType,
    depth: int,
    sibling_index: int,
    key: str | None,
) -> Generator[Node]:
    """Traverse a container type (dict, list, tuple, set)."""
    match node_type:
        case NodeType.DICT:
            items = value.items()
            count = len(value)
        case NodeType.SET:
            items = None
            count = len(value)
        case _:
            items = None
            count = len(value)

    yield Node(
        type=node_type,
        depth=depth,
        event=NodeEvent.ENTER,
        sibling_index=sibling_index,
        children_count=count,
        emptiness=count == 0,
        key=key,
    )

    if node_type == NodeType.DICT:
        for i, (k, v) in enumerate(items):
            # Yield the key as a leaf
            yield from traverse(k, _depth=depth + 1, _sibling_index=i)
            # Yield the value with the key attached
            yield from traverse(v, _depth=depth + 1, _sibling_index=i, _key=str(k))
    elif node_type == NodeType.SET:
        for i, item in enumerate(sorted(value, key=repr)):
            yield from traverse(item, _depth=depth + 1, _sibling_index=i)
    else:
        for i, item in enumerate(value):
            yield from traverse(item, _depth=depth + 1, _sibling_index=i)

    yield Node(
        type=node_type,
        depth=depth,
        event=NodeEvent.EXIT,
        sibling_index=sibling_index,
    )


def _make_leaf(
    value: object,
    node_type: NodeType,
    depth: int,
    sibling_index: int,
    key: str | None,
) -> Node:
    """Create a leaf node for a scalar value."""
    kwargs: dict = {}

    match node_type:
        case NodeType.INT | NodeType.FLOAT:
            kwargs["numeric_value"] = value
        case NodeType.STR:
            kwargs["string_length"] = len(value)
            kwargs["string_hash"] = _string_hash(value)
        case NodeType.BOOL:
            kwargs["bool_value"] = value
        case NodeType.NONE:
            pass
        case _:
            pass

    return Node(
        type=node_type,
        depth=depth,
        event=NodeEvent.LEAF,
        sibling_index=sibling_index,
        key=key,
        **kwargs,
    )
