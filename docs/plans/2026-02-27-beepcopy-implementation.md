# beepcopy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python module that copies data structures and sonifies their shape and values as audio, with a modular renderer system.

**Architecture:** Depth-first traversal produces a stream of Node objects with shape and value properties. Renderers consume nodes and emit AudioSegment descriptors. A numpy-based synthesizer turns segments into waveforms. Output handles playback and file export.

**Tech Stack:** Python 3.10+, numpy, wave (stdlib), simpleaudio (optional), uv, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/beepcopy/__init__.py`
- Create: `src/beepcopy/renderers/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "beepcopy"
version = "0.1.0"
description = "Listen to your data."
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
play = ["simpleaudio>=1.0.4"]
dev = ["pytest>=7.0", "simpleaudio>=1.0.4"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create empty package files**

`src/beepcopy/__init__.py`:
```python
"""beepcopy: Listen to your data."""
```

`src/beepcopy/renderers/__init__.py`:
```python
"""Sound renderers for beepcopy."""
```

`tests/__init__.py`: empty file

`tests/conftest.py`:
```python
"""Shared test fixtures for beepcopy."""
```

**Step 3: Initialize uv and install in dev mode**

Run: `uv init --no-readme && uv add --dev pytest && uv pip install -e ".[dev]"`

If `uv init` complains about existing pyproject.toml, skip it and just run:
`uv sync && uv pip install -e ".[dev]"`

**Step 4: Verify pytest runs**

Run: `uv run pytest -v`
Expected: "no tests ran" (0 collected), exit 5 (no tests), no import errors

**Step 5: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: scaffold beepcopy project structure"
```

---

### Task 2: Data Models (Node & AudioSegment)

**Files:**
- Create: `src/beepcopy/models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

`tests/test_models.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write minimal implementation**

`src/beepcopy/models.py`:
```python
"""Data models for beepcopy traversal and audio pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/beepcopy/models.py tests/test_models.py
git commit -m "feat: add Node and AudioSegment data models"
```

---

### Task 3: Data Structure Traverser

**Files:**
- Create: `src/beepcopy/traverser.py`
- Create: `tests/test_traverser.py`

**Step 1: Write the failing tests**

`tests/test_traverser.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_traverser.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

`src/beepcopy/traverser.py`:
```python
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
            items = value.items()  # type: ignore[union-attr]
            count = len(value)  # type: ignore[arg-type]
        case NodeType.SET:
            items = None
            count = len(value)  # type: ignore[arg-type]
        case _:
            items = None
            count = len(value)  # type: ignore[arg-type]

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
        for i, (k, v) in enumerate(items):  # type: ignore[arg-type]
            # Yield the key as a leaf
            yield from traverse(k, _depth=depth + 1, _sibling_index=i)
            # Yield the value with the key attached
            yield from traverse(v, _depth=depth + 1, _sibling_index=i, _key=str(k))
    elif node_type == NodeType.SET:
        for i, item in enumerate(sorted(value, key=repr)):  # type: ignore[arg-type]
            yield from traverse(item, _depth=depth + 1, _sibling_index=i)
    else:
        for i, item in enumerate(value):  # type: ignore[arg-type]
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
            kwargs["string_length"] = len(value)  # type: ignore[arg-type]
            kwargs["string_hash"] = _string_hash(value)  # type: ignore[arg-type]
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_traverser.py -v`
Expected: All 14 tests PASS

**Step 5: Commit**

```bash
git add src/beepcopy/traverser.py tests/test_traverser.py
git commit -m "feat: add depth-first data structure traverser"
```

---

### Task 4: Audio Synthesizer

**Files:**
- Create: `src/beepcopy/synthesizer.py`
- Create: `tests/test_synthesizer.py`

**Step 1: Write the failing tests**

`tests/test_synthesizer.py`:
```python
"""Tests for beepcopy audio synthesizer."""

import numpy as np

from beepcopy.models import AudioSegment, WaveShape
from beepcopy.synthesizer import synthesize, generate_waveform


class TestGenerateWaveform:
    """Test individual waveform generation."""

    def test_sine_wave_shape(self):
        samples = generate_waveform(WaveShape.SINE, 440.0, 0.01, 44100)
        assert isinstance(samples, np.ndarray)
        assert len(samples) == int(44100 * 0.01)
        # Sine wave should be in [-1, 1]
        assert samples.max() <= 1.0
        assert samples.min() >= -1.0

    def test_square_wave_values(self):
        samples = generate_waveform(WaveShape.SQUARE, 440.0, 0.01, 44100)
        # Square wave values should be close to -1 or 1
        unique_approx = set(np.sign(samples))
        assert unique_approx <= {-1.0, 0.0, 1.0}

    def test_sawtooth_range(self):
        samples = generate_waveform(WaveShape.SAWTOOTH, 440.0, 0.01, 44100)
        assert samples.max() <= 1.0
        assert samples.min() >= -1.0

    def test_triangle_range(self):
        samples = generate_waveform(WaveShape.TRIANGLE, 440.0, 0.01, 44100)
        assert samples.max() <= 1.0
        assert samples.min() >= -1.0

    def test_noise_is_random(self):
        s1 = generate_waveform(WaveShape.NOISE, 440.0, 0.1, 44100)
        s2 = generate_waveform(WaveShape.NOISE, 440.0, 0.1, 44100)
        # Two noise generations should not be identical
        assert not np.array_equal(s1, s2)


class TestSynthesize:
    """Test full synthesis pipeline from AudioSegments to buffer."""

    def test_single_segment(self):
        segments = [
            AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SINE),
        ]
        buffer = synthesize(segments, sample_rate=44100)
        assert isinstance(buffer, np.ndarray)
        expected_length = int(44100 * 0.1)
        assert abs(len(buffer) - expected_length) <= 1

    def test_amplitude_scaling(self):
        loud = [AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SINE, amplitude=1.0)]
        quiet = [AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SINE, amplitude=0.5)]
        loud_buf = synthesize(loud, sample_rate=44100)
        quiet_buf = synthesize(quiet, sample_rate=44100)
        # Quiet buffer should have smaller peak amplitude
        assert np.abs(quiet_buf).max() < np.abs(loud_buf).max()

    def test_sequential_segments(self):
        segments = [
            AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SINE, start_time=0.0),
            AudioSegment(frequency=880.0, duration=0.1, wave_shape=WaveShape.SINE, start_time=0.1),
        ]
        buffer = synthesize(segments, sample_rate=44100)
        expected_length = int(44100 * 0.2)
        assert abs(len(buffer) - expected_length) <= 1

    def test_overlapping_segments_mix(self):
        segments = [
            AudioSegment(frequency=440.0, duration=0.2, wave_shape=WaveShape.SINE, start_time=0.0),
            AudioSegment(frequency=880.0, duration=0.2, wave_shape=WaveShape.SINE, start_time=0.0),
        ]
        buffer = synthesize(segments, sample_rate=44100)
        # Buffer should be the length of the longest segment
        expected_length = int(44100 * 0.2)
        assert abs(len(buffer) - expected_length) <= 1

    def test_empty_segments(self):
        buffer = synthesize([], sample_rate=44100)
        assert len(buffer) == 0

    def test_envelope_no_clicks(self):
        """Start and end of audio should be near zero (envelope applied)."""
        segments = [
            AudioSegment(frequency=440.0, duration=0.1, wave_shape=WaveShape.SQUARE, amplitude=1.0),
        ]
        buffer = synthesize(segments, sample_rate=44100)
        # First and last few samples should be near zero due to envelope
        assert abs(buffer[0]) < 0.05
        assert abs(buffer[-1]) < 0.05
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_synthesizer.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

`src/beepcopy/synthesizer.py`:
```python
"""Audio synthesis from AudioSegment descriptors using numpy."""

from __future__ import annotations

import numpy as np

from beepcopy.models import AudioSegment, WaveShape

DEFAULT_SAMPLE_RATE = 44100


def generate_waveform(
    shape: WaveShape,
    frequency: float,
    duration: float,
    sample_rate: int,
) -> np.ndarray:
    """Generate a raw waveform of the given shape."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    match shape:
        case WaveShape.SINE:
            return np.sin(2 * np.pi * frequency * t)
        case WaveShape.SQUARE:
            return np.sign(np.sin(2 * np.pi * frequency * t))
        case WaveShape.SAWTOOTH:
            return 2 * (frequency * t % 1) - 1
        case WaveShape.TRIANGLE:
            return 2 * np.abs(2 * (frequency * t % 1) - 1) - 1
        case WaveShape.NOISE:
            rng = np.random.default_rng()
            return rng.uniform(-1.0, 1.0, num_samples)


def _apply_envelope(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply an attack/decay envelope to prevent clicks."""
    n = len(samples)
    if n == 0:
        return samples

    # 5ms attack and decay
    fade_samples = min(int(sample_rate * 0.005), n // 2)
    if fade_samples == 0:
        return samples

    envelope = np.ones(n)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return samples * envelope


def synthesize(
    segments: list[AudioSegment],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Synthesize a list of AudioSegments into a single audio buffer."""
    if not segments:
        return np.array([], dtype=np.float64)

    # Determine total buffer length
    end_times = [seg.start_time + seg.duration for seg in segments]
    total_duration = max(end_times)
    total_samples = int(sample_rate * total_duration)
    buffer = np.zeros(total_samples, dtype=np.float64)

    for seg in segments:
        waveform = generate_waveform(seg.wave_shape, seg.frequency, seg.duration, sample_rate)
        waveform = _apply_envelope(waveform, sample_rate)
        waveform *= seg.amplitude

        start_sample = int(seg.start_time * sample_rate)
        end_sample = start_sample + len(waveform)
        # Trim if it overflows
        if end_sample > total_samples:
            waveform = waveform[:total_samples - start_sample]
            end_sample = total_samples
        buffer[start_sample:end_sample] += waveform

    # Normalize to prevent clipping
    peak = np.abs(buffer).max()
    if peak > 1.0:
        buffer /= peak

    return buffer
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_synthesizer.py -v`
Expected: All 11 tests PASS

**Step 5: Commit**

```bash
git add src/beepcopy/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: add numpy-based audio synthesizer"
```

---

### Task 5: Audio Output (Playback & File Export)

**Files:**
- Create: `src/beepcopy/output.py`
- Create: `tests/test_output.py`

**Step 1: Write the failing tests**

`tests/test_output.py`:
```python
"""Tests for beepcopy audio output (file export and playback)."""

import wave
from pathlib import Path

import numpy as np

from beepcopy.output import write_wav, get_player


class TestWriteWav:
    def test_writes_valid_wav(self, tmp_path: Path):
        buffer = np.sin(np.linspace(0, 100, 4410))  # ~0.1s of audio
        output_path = tmp_path / "test.wav"
        write_wav(buffer, str(output_path), sample_rate=44100)

        assert output_path.exists()
        with wave.open(str(output_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit
            assert wf.getframerate() == 44100
            assert wf.getnframes() == len(buffer)

    def test_empty_buffer(self, tmp_path: Path):
        buffer = np.array([], dtype=np.float64)
        output_path = tmp_path / "empty.wav"
        write_wav(buffer, str(output_path), sample_rate=44100)
        assert output_path.exists()


class TestGetPlayer:
    def test_returns_callable(self):
        player = get_player()
        assert callable(player)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_output.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

`src/beepcopy/output.py`:
```python
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
        thread = threading.Thread(target=player, args=(buffer, sample_rate), daemon=True)
        thread.start()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_output.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/beepcopy/output.py tests/test_output.py
git commit -m "feat: add WAV file export and audio playback"
```

---

### Task 6: BaseRenderer & RetroRenderer

**Files:**
- Modify: `src/beepcopy/renderers/__init__.py`
- Create: `src/beepcopy/renderers/retro.py`
- Create: `tests/test_renderers.py`

**Step 1: Write the failing tests**

`tests/test_renderers.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderers.py -v`
Expected: FAIL with `ImportError`

**Step 3a: Write BaseRenderer**

`src/beepcopy/renderers/__init__.py`:
```python
"""Sound renderers for beepcopy."""

from __future__ import annotations

from beepcopy.models import AudioSegment, Node


class BaseRenderer:
    """Base class for all sound renderers.

    Subclass and override on_node() to create custom renderers.
    """

    def on_node(self, node: Node) -> list[AudioSegment]:
        """Given a traversal node, return audio segments to add.

        Override in subclasses.
        """
        return []

    def finalize(self) -> list[AudioSegment]:
        """Called after traversal. Returns all accumulated audio segments.

        Override in subclasses.
        """
        return []
```

**Step 3b: Write RetroRenderer**

`src/beepcopy/renderers/retro.py`:
```python
"""RetroRenderer: 8-bit sine/square wave beeps and boops."""

from __future__ import annotations

import math

from beepcopy.models import AudioSegment, Node, NodeEvent, NodeType, WaveShape
from beepcopy.renderers import BaseRenderer

# Map node types to waveforms for distinct sonic character
_TYPE_WAVEFORMS: dict[NodeType, WaveShape] = {
    NodeType.DICT: WaveShape.SQUARE,
    NodeType.LIST: WaveShape.SINE,
    NodeType.TUPLE: WaveShape.TRIANGLE,
    NodeType.SET: WaveShape.SAWTOOTH,
    NodeType.STR: WaveShape.SAWTOOTH,
    NodeType.INT: WaveShape.SQUARE,
    NodeType.FLOAT: WaveShape.SINE,
    NodeType.BOOL: WaveShape.TRIANGLE,
    NodeType.NONE: WaveShape.SINE,
    NodeType.OTHER: WaveShape.NOISE,
}

# Base frequencies for events (Hz)
_BASE_FREQ = 660.0
_ENTER_FREQ = 880.0
_EXIT_FREQ = 330.0


class RetroRenderer(BaseRenderer):
    """8-bit retro beeps. Depth controls pitch, types choose waveform, values modulate frequency."""

    def __init__(self, tempo: float = 120.0) -> None:
        self.tempo = tempo
        self._beat_duration = 60.0 / tempo  # duration of one beat in seconds
        self._note_duration = self._beat_duration * 0.25  # each node is a 16th note
        self._segments: list[AudioSegment] = []
        self._current_time: float = 0.0

    def on_node(self, node: Node) -> list[AudioSegment]:
        """Convert a node to retro beep segments."""
        freq = self._compute_frequency(node)
        wave = _TYPE_WAVEFORMS.get(node.type, WaveShape.SINE)
        amplitude = max(0.3, 1.0 - node.depth * 0.15)

        seg = AudioSegment(
            frequency=freq,
            duration=self._note_duration,
            wave_shape=wave,
            amplitude=amplitude,
            start_time=self._current_time,
        )
        self._segments.append(seg)
        self._current_time += self._note_duration
        return [seg]

    def finalize(self) -> list[AudioSegment]:
        """Return all accumulated segments."""
        return list(self._segments)

    def _compute_frequency(self, node: Node) -> float:
        """Compute frequency based on event type, depth, and value."""
        match node.event:
            case NodeEvent.ENTER:
                base = _ENTER_FREQ
            case NodeEvent.EXIT:
                base = _EXIT_FREQ
            case NodeEvent.LEAF:
                base = _BASE_FREQ

        # Depth lowers pitch: each level drops by a musical third (~1.26x)
        depth_factor = 1.0 / (1.26 ** node.depth)
        freq = base * depth_factor

        # Value modulation for leaves
        if node.event == NodeEvent.LEAF:
            if node.numeric_value is not None:
                # Map numeric value to semitone offset using log scale
                val = max(0.01, abs(node.numeric_value))
                semitones = math.log2(val) * 2  # 2 semitones per doubling
                freq *= 2 ** (semitones / 12)
            elif node.string_hash is not None:
                # Use hash to pick a note in the pentatonic scale
                pentatonic = [0, 2, 4, 7, 9]  # semitone offsets
                idx = node.string_hash % len(pentatonic)
                freq *= 2 ** (pentatonic[idx] / 12)

        # Clamp to audible range
        return max(80.0, min(freq, 4000.0))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_renderers.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/beepcopy/renderers/__init__.py src/beepcopy/renderers/retro.py tests/test_renderers.py
git commit -m "feat: add BaseRenderer and RetroRenderer"
```

---

### Task 7: Public API (beepcopy & beeplisten)

**Files:**
- Modify: `src/beepcopy/__init__.py`
- Create: `tests/test_api.py`

**Step 1: Write the failing tests**

`tests/test_api.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ImportError`

**Step 3: Write the public API**

`src/beepcopy/__init__.py`:
```python
"""beepcopy: Listen to your data."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from beepcopy.models import AudioSegment
from beepcopy.output import play, write_wav
from beepcopy.renderers import BaseRenderer
from beepcopy.renderers.retro import RetroRenderer
from beepcopy.synthesizer import synthesize
from beepcopy.traverser import traverse


def beepcopy(
    obj: Any,
    *,
    renderer: BaseRenderer | None = None,
    output: str | None = None,
    silent: bool = False,
    blocking: bool = False,
) -> Any:
    """Copy a data structure and sonify its shape and values.

    Args:
        obj: The data structure to copy.
        renderer: Sound renderer to use. Defaults to RetroRenderer.
        output: Output destination. None for speaker playback, a file path
                for WAV export, or "buffer" to return (copy, numpy_array).
        silent: If True, just copy without sound.
        blocking: If True, wait for playback to finish before returning.

    Returns:
        A deep copy of obj. If output="buffer", returns (copy, numpy_array).
    """
    copied = copy.deepcopy(obj)

    if not silent:
        buffer = _sonify(obj, renderer)

        if output == "buffer":
            return (copied, buffer)
        elif output is not None:
            write_wav(buffer, output)
        else:
            play(buffer, blocking=blocking)

    return copied


def beeplisten(
    obj: Any,
    *,
    renderer: BaseRenderer | None = None,
    output: str | None = None,
    silent: bool = False,
    blocking: bool = False,
) -> np.ndarray | None:
    """Listen to the shape and values of a data structure without copying.

    Args:
        obj: The data structure to sonify.
        renderer: Sound renderer to use. Defaults to RetroRenderer.
        output: Output destination. None for speaker playback, a file path
                for WAV export, or "buffer" to return the numpy array.
        silent: If True, do nothing (useful for toggling).
        blocking: If True, wait for playback to finish.

    Returns:
        None, or numpy array if output="buffer".
    """
    if silent:
        return None

    buffer = _sonify(obj, renderer)

    if output == "buffer":
        return buffer
    elif output is not None:
        write_wav(buffer, output)
    else:
        play(buffer, blocking=blocking)

    return None


def _sonify(obj: Any, renderer: BaseRenderer | None) -> np.ndarray:
    """Traverse a data structure and synthesize audio from it."""
    if renderer is None:
        renderer = RetroRenderer()

    for node in traverse(obj):
        renderer.on_node(node)

    segments = renderer.finalize()
    return synthesize(segments)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/beepcopy/__init__.py tests/test_api.py
git commit -m "feat: add beepcopy() and beeplisten() public API"
```

---

### Task 8: RhythmRenderer

**Files:**
- Create: `src/beepcopy/renderers/rhythm.py`
- Modify: `tests/test_renderers.py`

**Step 1: Write the failing tests**

Add to `tests/test_renderers.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderers.py::TestRhythmRenderer -v`
Expected: FAIL with `ImportError`

**Step 3: Write RhythmRenderer**

`src/beepcopy/renderers/rhythm.py`:
```python
"""RhythmRenderer: Percussive drum-pattern sonification."""

from __future__ import annotations

import math

from beepcopy.models import AudioSegment, Node, NodeEvent, NodeType, WaveShape
from beepcopy.renderers import BaseRenderer

# Kick drum: low sine
_KICK_FREQ = 60.0
# Snare: mid noise burst
_SNARE_FREQ = 200.0
# Hi-hat: high noise
_HIHAT_FREQ = 800.0


class RhythmRenderer(BaseRenderer):
    """Percussive renderer. Types map to drum sounds, values control velocity."""

    def __init__(self, tempo: float = 120.0) -> None:
        self.tempo = tempo
        self._beat_duration = 60.0 / tempo
        self._hit_duration = self._beat_duration * 0.2  # short percussive hits
        self._segments: list[AudioSegment] = []
        self._current_time: float = 0.0

    def on_node(self, node: Node) -> list[AudioSegment]:
        """Convert a node to a percussive hit."""
        freq, wave, base_amp = self._drum_sound(node)
        amplitude = base_amp * self._velocity(node)

        seg = AudioSegment(
            frequency=freq,
            duration=self._hit_duration,
            wave_shape=wave,
            amplitude=min(amplitude, 1.0),
            start_time=self._current_time,
        )
        self._segments.append(seg)
        self._current_time += self._hit_duration * 1.5  # slight gap between hits
        return [seg]

    def finalize(self) -> list[AudioSegment]:
        return list(self._segments)

    def _drum_sound(self, node: Node) -> tuple[float, WaveShape, float]:
        """Choose drum sound based on node type and event."""
        match (node.type, node.event):
            case (NodeType.DICT, NodeEvent.ENTER | NodeEvent.EXIT):
                return (_KICK_FREQ, WaveShape.SINE, 0.9)  # Kick
            case (NodeType.LIST, NodeEvent.ENTER | NodeEvent.EXIT):
                return (_HIHAT_FREQ, WaveShape.NOISE, 0.5)  # Hi-hat
            case (NodeType.TUPLE, NodeEvent.ENTER | NodeEvent.EXIT):
                return (_SNARE_FREQ, WaveShape.NOISE, 0.7)  # Snare
            case (NodeType.SET, NodeEvent.ENTER | NodeEvent.EXIT):
                return (400.0, WaveShape.TRIANGLE, 0.6)  # Rim shot
            case _:
                # Scalars get a snare-like sound
                return (_SNARE_FREQ, WaveShape.NOISE, 0.6)

    def _velocity(self, node: Node) -> float:
        """Compute hit velocity from node value."""
        if node.numeric_value is not None:
            val = max(0.01, abs(node.numeric_value))
            # Log scale: maps 1-1000 to ~0.5-1.0
            return min(1.0, 0.5 + math.log10(val) / 6)
        return 0.7  # default velocity
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_renderers.py -v`
Expected: All 15 tests PASS (10 previous + 5 new)

**Step 5: Commit**

```bash
git add src/beepcopy/renderers/rhythm.py tests/test_renderers.py
git commit -m "feat: add RhythmRenderer with percussive drum patterns"
```

---

### Task 9: TechnoRenderer

**Files:**
- Create: `src/beepcopy/renderers/techno.py`
- Modify: `tests/test_renderers.py`

**Step 1: Write the failing tests**

Add to `tests/test_renderers.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_renderers.py::TestTechnoRenderer -v`
Expected: FAIL with `ImportError`

**Step 3: Write TechnoRenderer**

`src/beepcopy/renderers/techno.py`:
```python
"""TechnoRenderer: EDM/techno-flavored data sonification."""

from __future__ import annotations

import math

from beepcopy.models import AudioSegment, Node, NodeEvent, NodeType, WaveShape
from beepcopy.renderers import BaseRenderer

# Acid bass note frequencies (A minor pentatonic)
_ACID_NOTES = [55.0, 65.4, 73.4, 82.4, 98.0, 110.0, 130.8, 146.8, 164.8, 196.0]

# Four-on-the-floor kick
_KICK_FREQ = 50.0


class TechnoRenderer(BaseRenderer):
    """EDM/techno renderer. Layered synths, arpeggios, acid bass, four-on-the-floor kick."""

    def __init__(self, tempo: float = 128.0) -> None:
        self.tempo = tempo
        self._beat_duration = 60.0 / tempo
        self._sixteenth = self._beat_duration * 0.25
        self._segments: list[AudioSegment] = []
        self._current_time: float = 0.0
        self._beat_count: int = 0

    def on_node(self, node: Node) -> list[AudioSegment]:
        """Convert a node to techno elements."""
        new_segments: list[AudioSegment] = []

        match node.event:
            case NodeEvent.ENTER:
                new_segments.extend(self._on_enter(node))
            case NodeEvent.LEAF:
                new_segments.extend(self._on_leaf(node))
            case NodeEvent.EXIT:
                new_segments.extend(self._on_exit(node))

        self._segments.extend(new_segments)
        self._current_time += self._sixteenth
        self._beat_count += 1
        return new_segments

    def finalize(self) -> list[AudioSegment]:
        return list(self._segments)

    def _on_enter(self, node: Node) -> list[AudioSegment]:
        """Container entry: kick + pad swell."""
        segments = []
        # Four-on-the-floor kick on container entries
        segments.append(AudioSegment(
            frequency=_KICK_FREQ,
            duration=self._sixteenth * 1.5,
            wave_shape=WaveShape.SINE,
            amplitude=0.9,
            start_time=self._current_time,
        ))
        # Pad layer: chord tone based on depth
        pad_freq = 220.0 * (2 ** (node.depth * 7 / 12))  # up a fifth per depth
        segments.append(AudioSegment(
            frequency=min(pad_freq, 2000.0),
            duration=self._sixteenth * 2,
            wave_shape=WaveShape.TRIANGLE,
            amplitude=0.3,
            start_time=self._current_time,
        ))
        return segments

    def _on_leaf(self, node: Node) -> list[AudioSegment]:
        """Scalar values: acid bass line + arpeggio notes."""
        segments = []

        # Acid bass from numeric values
        if node.numeric_value is not None:
            val = max(0.01, abs(node.numeric_value))
            note_idx = int(math.log2(max(val, 1)) * 2) % len(_ACID_NOTES)
            freq = _ACID_NOTES[note_idx]
            segments.append(AudioSegment(
                frequency=freq,
                duration=self._sixteenth * 0.8,
                wave_shape=WaveShape.SAWTOOTH,
                amplitude=0.7,
                start_time=self._current_time,
            ))
        elif node.string_hash is not None:
            # Strings get an arpeggiated synth
            note_idx = node.string_hash % len(_ACID_NOTES)
            freq = _ACID_NOTES[note_idx] * 2  # up an octave
            segments.append(AudioSegment(
                frequency=freq,
                duration=self._sixteenth * 0.6,
                wave_shape=WaveShape.SQUARE,
                amplitude=0.5,
                start_time=self._current_time,
            ))
        else:
            # Bool/None: hi-hat tick
            segments.append(AudioSegment(
                frequency=800.0,
                duration=self._sixteenth * 0.3,
                wave_shape=WaveShape.NOISE,
                amplitude=0.4,
                start_time=self._current_time,
            ))

        return segments

    def _on_exit(self, node: Node) -> list[AudioSegment]:
        """Container exit: crash/sweep."""
        return [AudioSegment(
            frequency=400.0 / (node.depth + 1),
            duration=self._sixteenth * 2,
            wave_shape=WaveShape.NOISE,
            amplitude=0.3,
            start_time=self._current_time,
        )]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_renderers.py -v`
Expected: All 19 tests PASS (15 previous + 4 new)

**Step 5: Commit**

```bash
git add src/beepcopy/renderers/techno.py tests/test_renderers.py
git commit -m "feat: add TechnoRenderer with EDM-style sonification"
```

---

### Task 10: Full Integration Test & README

**Files:**
- Create: `tests/test_integration.py`
- Modify: `README.md`

**Step 1: Write integration test**

`tests/test_integration.py`:
```python
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
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_integration.py -v`
Expected: All 8 tests PASS

**Step 3: Update README**

`README.md`:
```markdown
# beepcopy

Listen to your data.

A Python module that copies data structures and sonifies their shape and values as audio. Born from a typo of `deepcopy`.

## Install

```bash
pip install beepcopy

# With speaker playback support:
pip install beepcopy[play]
```

## Usage

```python
from beepcopy import beepcopy, beeplisten

# Copy and listen (drop-in alongside deepcopy)
data = {"users": [{"name": "Alice", "scores": [95, 87, 92]}]}
copied = beepcopy(data)

# Just listen, no copy
beeplisten(data)

# Export to file
beepcopy(data, output="my_data.wav")

# Silent copy (same as deepcopy)
copied = beepcopy(data, silent=True)
```

## Renderers

beepcopy ships with three sound styles:

```python
from beepcopy import beepcopy
from beepcopy.renderers.retro import RetroRenderer
from beepcopy.renderers.rhythm import RhythmRenderer
from beepcopy.renderers.techno import TechnoRenderer

data = [1, 2, 3, {"nested": True}]

beepcopy(data, renderer=RetroRenderer())   # 8-bit beeps and boops
beepcopy(data, renderer=RhythmRenderer())  # Percussive drum patterns
beepcopy(data, renderer=TechnoRenderer())  # EDM / acid techno
```

## Custom Renderers

```python
from beepcopy.renderers import BaseRenderer
from beepcopy.models import AudioSegment, WaveShape

class MyRenderer(BaseRenderer):
    def on_node(self, node):
        return [AudioSegment(
            frequency=440.0 * (node.depth + 1),
            duration=0.1,
            wave_shape=WaveShape.SINE,
        )]

    def finalize(self):
        return self._segments

beepcopy(data, renderer=MyRenderer())
```

## Requirements

- Python 3.10+
- numpy
- simpleaudio (optional, for speaker playback)
```

**Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS across all test files

**Step 5: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "feat: add integration tests and update README"
```

---

### Task 11: Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest -v --tb=short`
Expected: All tests PASS

**Step 2: Test manual playback (if simpleaudio installed)**

Run:
```bash
uv run python -c "
from beepcopy import beepcopy
data = {'hello': [1, 2, 3], 'world': {'nested': True}}
beepcopy(data, output='demo.wav')
print('Wrote demo.wav')
"
```
Expected: `demo.wav` file created, playable audio

**Step 3: Verify package installs clean**

Run: `uv pip install -e . && uv run python -c "from beepcopy import beepcopy; print('OK')"`
Expected: `OK`

**Step 4: Commit any remaining changes**

```bash
git status
# If anything remains:
git add -A && git commit -m "chore: final cleanup"
```
