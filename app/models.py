from datetime import datetime
from sys import displayhook

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class UserProfile(Base):
    """
    Stores agent-discovered user data.
    Wallet address is the primary key (Web3 identity).
    """

    __tablename__ = "user_profiles"

    wallet_address = Column(String(42), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    email = Column(String(255), nullable=True, unique=True)
    display_name = Column(String(100), nullable=True)

    career_context = Column(JSON, default=dict, nullable=True)
    skill_profile = Column(JSON, default=dict, nullable=True)
    learning_preferences = Column(JSON, default=dict, nullable=True)
    learning_challenges = Column(JSON, default=list, nullable=True)

    total_conversations = Column(Integer, default=0, nullable=False)
    last_active = Column(DateTime, nullable=True)

    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    recommendations = relationship(
        "CourseRecommendation", back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    """
    Stores all agent conversations for context and analysis.
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(
        String(42),
        ForeignKey("user_profiles.wallet_address", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    agent_type = Column(
        String(50), nullable=False
    )  # 'student', 'evaluation', 'content'
    mode = Column(
        String(50), nullable=True
    )  # For student agent: 'career', 'learning', 'progress', etc.

    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)

    # Context
    course_id = Column(
        Integer, nullable=True
    )  # If conversation was about a specific course
    chapter_id = Column(Integer, nullable=True)  # If discussing a specific chapter

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    tokens_used = Column(Integer, default=0)  # For cost tracking

    # Relationships
    user = relationship("UserProfile", back_populates="conversations")

    def __repr__(self):
        return f"<Conversation(id={self.id}, user={self.wallet_address}, role={self.role})>"


class CourseRecommendation(Base):
    """Conversation
    Stores agent-generated course recommendations.
    """

    __tablename__ = "course_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(
        String(42),
        ForeignKey("user_profiles.wallet_address", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    course_id = Column(Integer, nullable=False)
    reason = Column(Text, nullable=False)  # Why recommended
    priority = Column(Integer, default=3)  # 1=highest, 5=lowest

    # Status
    is_viewed = Column(Boolean, default=False)
    is_enrolled = Column(Boolean, default=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("UserProfile", back_populates="recommendations")


# ==================== AGENT ANALYTICS ====================


class AgentAnalytics(Base):
    """
    Track agent usage and performance metrics.
    """

    __tablename__ = "agent_analytics"

    id = Column(Integer, primary_key=True, autoincrement=True)

    agent_type = Column(
        String(50), nullable=False, index=True
    )  # 'student', 'evaluation', 'content'
    event_type = Column(
        String(50), nullable=False
    )  # 'chat', 'recommendation', 'evaluation', etc.

    # Performance metrics
    execution_time_ms = Column(Integer, default=0)
    tokens_used = Column(Integer, default=0)
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    # Context
    wallet_address = Column(String(42), nullable=True)
    course_id = Column(Integer, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


Index(
    "idx_conversations_user_date", Conversation.wallet_address, Conversation.created_at
)
Index(
    "idx_recommendations_user_priority",
    CourseRecommendation.wallet_address,
    CourseRecommendation.priority,
)
Index("idx_analytics_agent_date", AgentAnalytics.agent_type, AgentAnalytics.created_at)
