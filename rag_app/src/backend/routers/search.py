from fastapi import APIRouter
from rag_app.backend.schemas.search import SearchHit, SearchRequest, SearchResponse

router = APIRouter()


@router.post("/", response_model=SearchResponse, summary="Search (placeholder)")
async def search(req: SearchRequest) -> SearchResponse:
    # TODO: wire to core retrieval (lexical/vector/hybrid)
    demo = SearchHit(filename="README.md", score=1.0, title="Demo", content="Placeholder result")
    return SearchResponse(results=[demo])
