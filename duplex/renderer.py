"""
Renderer: interprets special action tokens in model output to produce clean text.

Handles:
  <|REVISE_START|> ... <|REVISE_END|> — replaces the most recent matching span
  <|INSERT|> text — inserts text at current position

The renderer maintains the visible output and applies revisions in-order.
"""

from duplex.config import SPECIAL_TOKENS

REVISE_START = SPECIAL_TOKENS["revise_start"]
REVISE_END = SPECIAL_TOKENS["revise_end"]
INSERT = SPECIAL_TOKENS["insert"]


class StreamRenderer:
    """Processes a stream of raw model output and yields clean rendered text."""

    def __init__(self):
        self.rendered = ""
        self._in_revision = False
        self._revision_buffer = ""

    def reset(self):
        self.rendered = ""
        self._in_revision = False
        self._revision_buffer = ""

    def feed(self, raw_text: str) -> str:
        """
        Feed raw model output (may contain action tokens) and return
        the current clean rendered text.
        """
        self.rendered = ""
        self._in_revision = False
        self._revision_buffer = ""

        i = 0
        while i < len(raw_text):
            if raw_text[i:].startswith(REVISE_START):
                self._in_revision = True
                self._revision_buffer = ""
                i += len(REVISE_START)
            elif raw_text[i:].startswith(REVISE_END):
                if self._in_revision:
                    self.rendered += self._revision_buffer
                    self._in_revision = False
                    self._revision_buffer = ""
                i += len(REVISE_END)
            elif raw_text[i:].startswith(INSERT):
                i += len(INSERT)
            else:
                if self._in_revision:
                    self._revision_buffer += raw_text[i]
                else:
                    self.rendered += raw_text[i]
                i += 1

        # If still in revision block (incomplete), show what we have so far
        if self._in_revision:
            self.rendered += self._revision_buffer

        return self.rendered


def render_final(raw_text: str) -> str:
    """One-shot render of a complete model output string."""
    renderer = StreamRenderer()
    return renderer.feed(raw_text)


def strip_action_tokens(text: str) -> str:
    """Remove all action tokens from text without applying revision logic."""
    for tok in SPECIAL_TOKENS.values():
        text = text.replace(tok, "")
    return text.strip()
