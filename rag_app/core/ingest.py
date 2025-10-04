"""Ingestion utilities.

Download and parse repository content (e.g., markdown/MDX) into records.
Wire actual logic later; keep API stable for backend/services.
"""

from typing import List, Dict


def load_docs() -> List[Dict]:
    """Placeholder loader returning an empty corpus.
    TODO: implement GitHub ZIP download + frontmatter parsing.
    """
    return []
