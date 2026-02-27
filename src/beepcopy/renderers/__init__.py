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
