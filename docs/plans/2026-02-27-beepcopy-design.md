# beepcopy Design Document

*Listen to your data.*

Born from a typo of Python's `deepcopy`, beepcopy is a Python module that copies data structures and sonifies their shape and values as audio. Fun, educational, and delightful.

## Core API

```python
from beepcopy import beepcopy, beeplisten

# Copy and play sound (drop-in alongside deepcopy)
copied = beepcopy(data)

# Silent copy
copied = beepcopy(data, silent=True)

# Export to WAV file
copied = beepcopy(data, output="structure.wav")

# Custom renderer
from beepcopy.renderers import RetroRenderer
copied = beepcopy(data, renderer=RetroRenderer(tempo=120))

# Just listen, no copy
beeplisten(data)
```

`beepcopy()` is the primary function. It walks the data structure, builds audio via a renderer, plays it (or exports), and returns a deep copy. `beeplisten()` is a convenience for sound-only use.

## Data Traversal & Node Model

The traverser walks any data structure depth-first and produces a stream of `Node` objects. Each node carries two layers of information:

### Shape properties (structural)

- **type**: `dict`, `list`, `tuple`, `set`, `str`, `int`, `float`, `bool`, `None`, `"other"`
- **depth**: nesting level (0 = root)
- **children_count**: number of children (containers only)
- **sibling_index**: position among siblings
- **event**: `enter` (entering a container), `leaf` (a scalar value), `exit` (leaving a container)

### Value properties (content-aware)

- **numeric_value**: actual number for `int`/`float`
- **string_length**: length for strings
- **string_hash**: deterministic hash of string content (same string always sounds the same)
- **bool_value**: `True`/`False`
- **emptiness**: whether a container is empty
- **key**: dict key if this node is a dict value

### Example

`{"a": [1, 2]}` produces:

```
enter dict  depth=0  children=1
  leaf str  depth=1  string_length=1  key=None    # key "a"
  enter list  depth=1  children=2  key="a"
    leaf int  depth=2  numeric_value=1
    leaf int  depth=2  numeric_value=2
  exit list  depth=1
exit dict  depth=0
```

Renderers decide which properties to use. This separation means traversal logic never changes -- only renderers do.

## Renderer System

Renderers implement a simple protocol:

```python
class BaseRenderer:
    def on_node(self, node: Node) -> list[AudioSegment]:
        """Given a traversal node, return audio segments."""
        ...

    def finalize(self) -> AudioBuffer:
        """Called after traversal. Returns the complete audio."""
        ...
```

### Built-in renderers

1. **RetroRenderer** (default) -- 8-bit sine/square wave beeps. Depth controls pitch (deeper = lower). Types choose waveform. Values modulate frequency.

2. **RhythmRenderer** -- Percussive. Types map to drum sounds (kick for dicts, hi-hat for lists, snare for scalars). Values control hit velocity. Structure creates rhythmic patterns.

3. **TechnoRenderer** -- EDM/techno. Layered synth pads, arpeggiated sequences from list values, acid bass from numeric ranges, four-on-the-floor kick from container entries.

### Custom renderers

```python
from beepcopy.renderers import BaseRenderer

class MyRenderer(BaseRenderer):
    def on_node(self, node):
        # Your sound logic
        ...

beepcopy(data, renderer=MyRenderer())
```

## Audio Pipeline

```
Data Structure -> Traverser -> [Nodes] -> Renderer -> [AudioSegments] -> Synthesizer -> AudioBuffer -> Output
```

**Synthesizer** takes `AudioSegment` descriptors and produces waveform data via numpy:
- Waveform generation: sine, square, sawtooth, triangle, noise
- Amplitude envelopes (attack/decay to prevent clicks)
- Mixing overlapping segments
- Default sample rate: 44100 Hz

**Output options:**
- **Direct playback** (default): `simpleaudio` if installed, else temp WAV + system player
- **File export**: `beepcopy(data, output="file.wav")`
- **Raw buffer**: `beepcopy(data, output="buffer")` returns numpy array

Playback is non-blocking by default (background thread). Pass `blocking=True` to wait.

## Project Structure

```
beepcopy/
  src/
    beepcopy/
      __init__.py          # beepcopy(), beeplisten() public API
      traverser.py         # Data structure walker, produces Node stream
      models.py            # Node, AudioSegment dataclasses
      synthesizer.py       # numpy waveform generation & mixing
      output.py            # Playback & file export
      renderers/
        __init__.py        # BaseRenderer, renderer registry
        retro.py           # RetroRenderer (default)
        rhythm.py          # RhythmRenderer
        techno.py          # TechnoRenderer
  tests/
    test_traverser.py
    test_synthesizer.py
    test_renderers.py
    test_api.py
  pyproject.toml
  README.md
  LICENSE
```

## Dependencies

- **Required**: `numpy`
- **Optional**: `simpleaudio` (`pip install beepcopy[play]`)
- **Python**: 3.10+
- **Dev tooling**: `uv`

## Technical Approach

- Audio synthesis via `numpy` -- raw waveform math, no audio framework
- File output via stdlib `wave` module
- Strategy pattern for renderers with a base class protocol
- Depth-first traversal with enter/leaf/exit events
