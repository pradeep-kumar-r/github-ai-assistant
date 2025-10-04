from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="Health check")
async def health():
    return {"status": "ok"}
