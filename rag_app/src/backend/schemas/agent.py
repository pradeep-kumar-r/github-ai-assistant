from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    conversation_id: Optional[str] = None
    # Optional: add references/citations later
