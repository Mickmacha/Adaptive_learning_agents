from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

# ==================== STUDENT AGENT ====================


class LearningContext(BaseModel):
    current_course_id: Optional[int] = None
    current_chapter: Optional[int] = None
    current_chapter_title: Optional[str] = None
    current_chapter_summary: Optional[str] = None
    # Use default_factory to avoid shared mutable default
    completed_courses: List[Dict] = Field(default_factory=list)


class StudentChatRequest(BaseModel):
    wallet_address: str = Field(
        ...,
        min_length=42,
        max_length=42,
        description="Ethereum wallet address",
    )
    message: str = Field(..., min_length=1, max_length=2000)
    # Optional top-level course id for convenience/backwards-compat
    current_course_id: Optional[int] = None
    # Frontend-supplied learning context snapshot
    learning_context: Optional[LearningContext] = None

    class Config:
        json_schema_extra = {
            "example": {
                "wallet_address": "0x1234567890123456789012345678901234567890",
                "message": "Help me understand smart contract security",
                "current_course_id": 5,
                "learning_context": {
                    "current_course_id": 5,
                    "current_chapter": 2,
                    "current_chapter_title": "Smart contract security basics",
                    "current_chapter_summary": "Overview of key vulnerabilities and best practices.",
                    "completed_courses": [
                        {"course_id": 1, "title": "Intro to Web3"}
                    ],
                },
            }
        }


class StudentChatResponse(BaseModel):
    response: str
    mode: str  # 'career', 'learning', 'progress', 'recommendation', 'general'
    profile_updated: bool
    recommendations: Optional[List[Dict]] = None  # If mode='recommendation'

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Smart contract security is crucial! Let me explain the key concepts...",
                "mode": "learning",
                "profile_updated": True,
                "recommendations": None,
            }
        }

# ==================== CAREER ONBOARDING ====================

class CareerOnboardingRequest(BaseModel):
    """Career onboarding form data structure."""
    currentStatus: str = Field(..., description="Current employment status")
    currentRole: Optional[str] = None
    yearsOfExperience: Optional[str] = None
    industryBackground: str
    technicalLevel: str
    programmingLanguages: List[str] = Field(default_factory=list)
    hasBlockchainExp: str
    hasAIExp: str
    targetRole: List[str] = Field(..., min_items=1, description="At least one target role required")
    careerTimeline: str
    geographicPreference: str
    primaryMotivation: List[str] = Field(default_factory=list)
    webThreeInterest: Optional[str] = None
    aiInterest: Optional[str] = None
    strongSkills: List[str] = Field(default_factory=list)
    wantToImprove: List[str] = Field(default_factory=list)
    learningStyle: str
    timeCommitment: str
    shortTermGoal: str
    concerns: Optional[str] = None
    additionalInfo: Optional[str] = None
    agreeToTerms: bool = Field(..., description="Must be true to submit")
    submittedAt: str  # ISO 8601 timestamp
    walletAddress: str = Field(..., min_length=42, max_length=42, description="Ethereum wallet address")

    class Config:
        json_schema_extra = {
            "example": {
                "currentStatus": "employed",
                "currentRole": "Software Engineer",
                "yearsOfExperience": "3",
                "industryBackground": "tech",
                "technicalLevel": "intermediate",
                "programmingLanguages": ["JavaScript/TypeScript", "Python"],
                "hasBlockchainExp": "minimal",
                "hasAIExp": "hands-on",
                "targetRole": ["Smart Contract Developer"],
                "careerTimeline": "6-12",
                "geographicPreference": "remote",
                "primaryMotivation": ["Learning new technologies"],
                "webThreeInterest": "defi",
                "aiInterest": "llm-apps",
                "strongSkills": ["Problem-solving", "Fast learner"],
                "wantToImprove": ["Technical skills"],
                "learningStyle": "hands-on",
                "timeCommitment": "10-15",
                "shortTermGoal": "portfolio",
                "concerns": "Finding time to learn",
                "additionalInfo": "I have React experience",
                "agreeToTerms": True,
                "submittedAt": "2024-01-15T10:30:00.000Z",
                "walletAddress": "0x1234567890123456789012345678901234567890"
            }
        }

class CareerOnboardingResponse(BaseModel):
    """Response after processing career onboarding."""
    success: bool
    profileId: str  # wallet_address
    recommendations: Dict
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "profileId": "0x1234567890123456789012345678901234567890",
                "recommendations": {
                    "learningPath": "Beginner → Intermediate → Advanced",
                    "recommendedCourses": ["Blockchain Fundamentals", "Solidity Basics"],
                    "skillPriorities": ["Smart Contracts", "DeFi"],
                    "timeline": "6-12"
                },
                "message": "Welcome! Your career profile has been created. Let's start your learning journey!"
            }
        }

# ==================== USER PROFILE ====================

class UserProfileResponse(BaseModel):
    wallet_address: str
    career_context: Dict
    skill_profile: Dict
    learning_preferences: Dict
    learning_challenges: List[str]
    total_conversations: int
    last_active: datetime

class UserProfileUpdate(BaseModel):
    career_context: Optional[Dict] = None
    skill_profile: Optional[Dict] = None
    learning_preferences: Optional[Dict] = None
    learning_challenges: Optional[List[str]] = None
    
# ==================== COURSE RECOMMENDATIONS ====================

class CourseRecommendationCreate(BaseModel):
    wallet_address: str
    course_id: int
    reason: str
    priority: int = Field(default=3, ge=1, le=5)

class CourseRecommendationResponse(BaseModel):
    id: int
    course_id: int
    reason: str
    priority: int
    is_viewed: bool
    is_enrolled: bool
    created_at: datetime

# ==================== ANALYTICS ====================

class AgentAnalyticsCreate(BaseModel):
    agent_type: str
    event_type: str
    execution_time_ms: int
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    wallet_address: Optional[str] = None
    course_id: Optional[int] = None

class AgentAnalyticsResponse(BaseModel):
    total_requests: int
    avg_execution_time_ms: float
    avg_tokens_per_request: float
    success_rate: float
    most_common_errors: List[Dict]