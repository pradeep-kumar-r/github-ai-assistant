from pydantic import BaseModel
from typing import Optional, List


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchHit(BaseModel):
    filename: str
    score: float
    title: Optional[str] = None
    content: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchHit]
