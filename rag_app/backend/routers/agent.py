from fastapi import APIRouter
from rag_app.backend.schemas.agent import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse, summary="Agent ask (placeholder)")
async def ask(req: AskRequest) -> AskResponse:
    # TODO: wire to core agent run
    return AskResponse(answer="This is a placeholder answer.", conversation_id=req.conversation_id)
