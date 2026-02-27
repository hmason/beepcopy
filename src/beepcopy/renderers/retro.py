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
