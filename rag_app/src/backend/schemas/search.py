
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = 5


class SearchHit(BaseModel):
    filename: str
    score: float
    title: str | None = None
    content: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchHit]
