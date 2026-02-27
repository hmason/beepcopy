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
