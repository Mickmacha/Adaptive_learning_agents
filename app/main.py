from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import get_db
from .schemas import (
    AgentAnalyticsCreate,
    CourseRecommendationCreate,
    CourseRecommendationResponse,
    StudentChatRequest,
    StudentChatResponse,
    UserProfileResponse,
    UserProfileUpdate,
)


app = FastAPI(
    title="Adaptive Learning Agents", description="API for Adaptive Learning Agents"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
