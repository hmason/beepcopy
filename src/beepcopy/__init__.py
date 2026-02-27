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
