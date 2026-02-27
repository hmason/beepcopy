"""Data models for beepcopy traversal and audio pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Type of data encountered during traversal."""
    DICT = "dict"
    LIST = "list"
    TUPLE = "tuple"
    SET = "set"
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    NONE = "none"
    OTHER = "other"


class NodeEvent(Enum):
    """Event type during traversal."""
    ENTER = "enter"
    LEAF = "leaf"
    EXIT = "exit"


class WaveShape(Enum):
    """Waveform shape for audio synthesis."""
    SINE = "sine"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"
    NOISE = "noise"


@dataclass(frozen=True)
class Node:
    """A single node in the traversal stream.

    Carries both shape (structural) and value (content-aware) properties.
    """
    # Shape properties
    type: NodeType
    depth: int
    event: NodeEvent
    sibling_index: int
    children_count: int | None = None

    # Value properties
    numeric_value: float | int | None = None
    string_length: int | None = None
    string_hash: int | None = None
    bool_value: bool | None = None
    emptiness: bool | None = None
    key: str | None = None


@dataclass(frozen=True)
class AudioSegment:
    """Descriptor for a sound to be synthesized.

    Renderers produce these; the synthesizer turns them into waveforms.
    """
    frequency: float
    duration: float
    wave_shape: WaveShape
    amplitude: float = 1.0
    start_time: float = 0.0
