
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    conversation_id: str | None = None
    # Optional: add references/citations later
