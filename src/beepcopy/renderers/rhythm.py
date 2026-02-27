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
