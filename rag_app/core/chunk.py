"""Chunking utilities.

Provide sliding-window and section-based splitting helpers.
"""

from typing import List, Dict


def sliding_window(text: str, size: int = 2000, step: int = 1000) -> List[Dict]:
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")
    n = len(text)
    chunks = []
    for i in range(0, n, step):
        chunk = text[i : i + size]
        chunks.append({"start": i, "content": chunk})
        if i + size >= n:
            break
    return chunks
