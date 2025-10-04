"""Retrieval orchestration layer.

Wire lexical (MinSearch) and optional vector backends here.
Keep this file thin and delegate heavy lifting to rag_app.core.*
"""

from typing import List, Dict


async def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    """Placeholder hybrid search combining lexical and vector results.
    TODO: implement using rag_app.core.search_lexical and optional vector backend.
    """
    return [
        {
            "filename": "README.md",
            "score": 1.0,
            "title": "Demo",
            "content": f"Placeholder result for: {query}",
        }
    ][: top_k or 5]
