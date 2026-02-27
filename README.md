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
    def __init__(self):
        self._segments = []

    def on_node(self, node):
        seg = AudioSegment(
            frequency=440.0 * (node.depth + 1),
            duration=0.1,
            wave_shape=WaveShape.SINE,
        )
        self._segments.append(seg)
        return [seg]

    def finalize(self):
        return self._segments

beepcopy(data, renderer=MyRenderer())
```

## Requirements

- Python 3.10+
- numpy
- simpleaudio (optional, for speaker playback)
