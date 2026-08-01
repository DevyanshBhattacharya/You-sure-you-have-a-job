"""Splitting email text into embeddable chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Rough char-per-token ratio for English prose. Good enough for sizing; we are
# not enforcing a hard model limit here, just keeping chunks retrievable.
CHARS_PER_TOKEN = 4
TARGET_TOKENS = 800
OVERLAP_TOKENS = 100

TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN

_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(slots=True)
class Chunk:
    index: int
    text: str
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_oversized(block: str) -> list[str]:
    """Break a single huge paragraph on sentence boundaries, then hard-cut."""
    if len(block) <= TARGET_CHARS:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+", block)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > TARGET_CHARS:
            if current:
                out.append(current)
                current = ""
            for i in range(0, len(sentence), TARGET_CHARS):
                out.append(sentence[i : i + TARGET_CHARS])
            continue
        if len(current) + len(sentence) + 1 > TARGET_CHARS:
            out.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return out


def chunk_text(text: str, *, header: str | None = None) -> list[Chunk]:
    """Split text into overlapping chunks, prefixing each with `header`.

    The header (sender/subject/date) is repeated on every chunk so a retrieved
    fragment still carries the context needed to interpret it.
    """
    body = (text or "").strip()
    if not body:
        return []

    blocks: list[str] = []
    for para in _PARAGRAPH.split(body):
        para = para.strip()
        if para:
            blocks.extend(_split_oversized(para))

    if not blocks:
        return []

    pieces: list[str] = []
    current = ""
    for block in blocks:
        if not current:
            current = block
        elif len(current) + len(block) + 2 <= TARGET_CHARS:
            current = f"{current}\n\n{block}"
        else:
            pieces.append(current)
            tail = current[-OVERLAP_CHARS:] if len(current) > OVERLAP_CHARS else current
            current = f"{tail}\n\n{block}"
    if current:
        pieces.append(current)

    prefix = f"{header.strip()}\n\n" if header else ""
    return [
        Chunk(index=i, text=prefix + piece, token_estimate=estimate_tokens(prefix + piece))
        for i, piece in enumerate(pieces)
    ]
