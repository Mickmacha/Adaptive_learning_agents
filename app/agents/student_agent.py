from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic
import operator
import json

from app.database import (
    get_user_profile,
    update_user_profile,
    save_conversation,
    get_conversation_history,
)
from app.config import settings


class StudentState(TypedDict):
    wallet_address: str
    # Loaded context from DB
    user_profile: dict
    conversation_history: Annotated[list, operator.add]
    conversation_summary: str  # Summarized context for long sessions

    # Frontend-supplied context (no Web3 calls needed)
    completed_courses: list | None
    current_course_id: int | None
    current_chapter: int | None
    current_chapter_title: str | None
    current_chapter_summary: str | None

    # Current interaction
    last_message: str
    mode: str  # 'career', 'learning', 'progress', 'recommendation', 'general', 'onboarding'
    
    # Onboarding form data (if present)
    onboarding_data: dict | None

    # Response data
    response: str
    profile_updates: dict


class StudentCompanionAgent:
    """Unified agent for all learner interactions."""

    def __init__(self, db_session):
        self.db = db_session
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", api_key=settings.GEMINI_API_KEY
        )
        self.graph = self.build_graph()

    def build_graph(self):
        """Build and compile the student companion workflow graph."""
        workflow = StateGraph(StudentState)

        # Add nodes
        workflow.add_node("load_context", self.load_context)
        workflow.add_node("summarize_if_needed", self.summarize_conversation_if_needed)
        workflow.add_node("determine_mode", self.determine_mode)
        workflow.add_node("career_mode", self.career_guidance)
        workflow.add_node("onboarding_mode", self.handle_onboarding)
        workflow.add_node("learning_mode", self.learning_assistance)
        workflow.add_node("progress_mode", self.progress_review)
        workflow.add_node("recommend_mode", self.course_recommendation)
        workflow.add_node("general_mode", self.general_conversation)
        workflow.add_node("update_profile", self.update_user_profile_node)

        # Entry point
        workflow.set_entry_point("load_context")

        # Sequential flow with summarization check
        workflow.add_edge("load_context", "summarize_if_needed")
        workflow.add_edge("summarize_if_needed", "determine_mode")

        # Conditional routing based on detected mode
        workflow.add_conditional_edges(
            "determine_mode",
            lambda s: s["mode"],
            {
                "onboarding": "onboarding_mode",
                "career": "career_mode",
                "learning": "learning_mode",
                "progress": "progress_mode",
                "recommendation": "recommend_mode",
                "general": "general_mode",
            },
        )

        # All modes converge to profile update
        workflow.add_edge("onboarding_mode", "update_profile")
        workflow.add_edge("career_mode", "update_profile")
        workflow.add_edge("learning_mode", "update_profile")
        workflow.add_edge("progress_mode", "update_profile")
        workflow.add_edge("recommend_mode", "update_profile")
        workflow.add_edge("general_mode", "update_profile")

        workflow.add_edge("update_profile", END)

        return workflow.compile()

    def load_context(self, state: StudentState) -> StudentState:
        """
        Load user context from the agent database (profile and recent history).

        All course/progress information is expected to be already present
        in the incoming state from the frontend.
        """
        wallet = state["wallet_address"]

        profile = get_user_profile(self.db, wallet) or {}
        history = get_conversation_history(self.db, wallet, limit=10)

        # Ensure wallet_address is stored (create profile if doesn't exist)
        if not profile:
            from app.database import create_user_profile
            create_user_profile(self.db, wallet)
            profile = get_user_profile(self.db, wallet) or {}

        return {
            **state,
            "user_profile": profile,
            "conversation_history": history,
            "conversation_summary": state.get("conversation_summary", ""),
            "profile_updates": {},
        }
            
    def determine_mode(self, state: StudentState) -> StudentState:
        """Decide what mode to operate in."""

        message = state["last_message"].lower()
        
        # Check if this is an onboarding form submission
        if state.get("onboarding_data"):
            mode = "onboarding"
        
        # Career onboarding for new users (conversational)
        elif not state.get("completed_courses") and not state["user_profile"].get(
            "career_context"
        ):
            mode = "career"

        # Learning help if user is in a course
        elif state["current_course_id"]:
            # Check if they're asking about progress or recommendations
            if any(kw in message for kw in ["progress", "how am i doing", "stats"]):
                mode = "progress"
            elif any(
                kw in message for kw in ["what next", "recommend", "what should i"]
            ):
                mode = "recommendation"
            elif any(kw in message for kw in ["what should i learn", "what course"]):
                mode = "recommendation"

            else:
                mode = "learning"

        # Progress review
        elif any(kw in message for kw in ["progress", "completed", "achievement"]):
            mode = "progress"

        # Career guidance
        elif any(kw in message for kw in ["career", "job", "become a", "work as"]):
            mode = "career"

        # Default
        else:
            mode = "general"

        return {**state, "mode": mode}
    
    def summarize_conversation_if_needed(self, state: StudentState) -> StudentState:
        """
        Summarize conversation history if it gets too long (memory management).
        Keeps recent messages but summarizes older ones to save tokens.
        """
        history = state.get("conversation_history", [])
        summary = state.get("conversation_summary", "")
        
        # If we have more than 15 messages, summarize the older ones
        if len(history) > 15:
            # Keep last 5 messages, summarize the rest
            recent = history[-5:]
            to_summarize = history[:-5]
            
            if to_summarize:
                summary_prompt = f"""Summarize this conversation history into a concise summary 
that captures key information about the user's goals, progress, and preferences:

Previous summary: {summary if summary else "None"}

Conversation to summarize:
{json.dumps(to_summarize, indent=2)}

Return a brief summary (2-3 sentences) that preserves:
- User's career goals and target roles
- Learning progress and completed courses
- Key preferences and challenges
- Important context for future conversations"""
                
                summary_response = self.llm.invoke([{"role": "user", "content": summary_prompt}])
                new_summary = summary_response.content
                
                return {
                    **state,
                    "conversation_history": recent,
                    "conversation_summary": new_summary,
                }
        
        return state
    
    def handle_onboarding(self, state: StudentState) -> StudentState:
        """
        Handle career onboarding form submission.
        Processes structured form data and creates initial career profile.
        """
        onboarding_data = state.get("onboarding_data")
        if not onboarding_data:
            # Fallback to conversational onboarding if no form data
            return self.career_guidance(state)
        
        wallet = state["wallet_address"]
        
        # Transform form data into structured career context
        career_context = {
            "current_status": onboarding_data.get("currentStatus"),
            "current_role": onboarding_data.get("currentRole"),
            "years_of_experience": onboarding_data.get("yearsOfExperience"),
            "industry_background": onboarding_data.get("industryBackground"),
            "technical_level": onboarding_data.get("technicalLevel"),
            "programming_languages": onboarding_data.get("programmingLanguages", []),
            "blockchain_experience": onboarding_data.get("hasBlockchainExp"),
            "ai_experience": onboarding_data.get("hasAIExp"),
            "target_role": onboarding_data.get("targetRole", []),
            "career_timeline": onboarding_data.get("careerTimeline"),
            "geographic_preference": onboarding_data.get("geographicPreference"),
            "primary_motivation": onboarding_data.get("primaryMotivation", []),
            "web3_interest": onboarding_data.get("webThreeInterest"),
            "ai_interest": onboarding_data.get("aiInterest"),
            "onboarding_completed_at": onboarding_data.get("submittedAt"),
        }
        
        learning_preferences = {
            "learning_style": onboarding_data.get("learningStyle"),
            "time_commitment": onboarding_data.get("timeCommitment"),
            "short_term_goal": onboarding_data.get("shortTermGoal"),
            "concerns": onboarding_data.get("concerns"),
        }
        
        skill_profile = {
            "strong_skills": onboarding_data.get("strongSkills", []),
            "want_to_improve": onboarding_data.get("wantToImprove", []),
        }
        
        # Generate personalized recommendations using LLM
        profile_summary = f"""New user onboarding:
- Status: {career_context['current_status']}
- Experience: {career_context.get('years_of_experience', 'N/A')} years, {career_context['technical_level']} level
- Skills: {', '.join(career_context.get('programming_languages', []))}
- Blockchain: {career_context['blockchain_experience']}, AI: {career_context['ai_experience']}
- Goals: {', '.join(career_context['target_role'])} within {career_context['career_timeline']} months
- Learning: {learning_preferences['learning_style']}, {learning_preferences['time_commitment']} hrs/week
- Short-term: {learning_preferences['short_term_goal']}
- Interests: Web3={career_context.get('web3_interest', 'general')}, AI={career_context.get('ai_interest', 'general')}"""
        
        recommendation_prompt = f"""You are a career advisor for a Web3/AI learning platform.

{profile_summary}

Generate personalized recommendations:
1. A learning path (e.g., "Beginner → Intermediate → Advanced")
2. Top 3-5 recommended course topics/modules they should start with
3. Key skill priorities to focus on
4. Timeline alignment with their {career_context['career_timeline']}-month goal

Return as JSON:
{{
    "learningPath": "string",
    "recommendedCourses": ["course1", "course2", ...],
    "skillPriorities": ["skill1", "skill2", ...],
    "timeline": "{career_context['career_timeline']}",
    "welcomeMessage": "Personalized welcome message"
}}"""
        
        recommendation_response = self.llm.invoke([{"role": "user", "content": recommendation_prompt}])
        
        try:
            recommendations = json.loads(
                recommendation_response.content.strip("```json").strip("```")
            )
        except Exception:
            recommendations = {
                "learningPath": "Beginner → Intermediate → Advanced",
                "recommendedCourses": ["Blockchain Fundamentals", "Smart Contract Development"],
                "skillPriorities": ["Solidity", "Web3 Architecture"],
                "timeline": career_context["career_timeline"],
                "welcomeMessage": "Welcome! Let's start your Web3 learning journey!",
            }
        
        # Prepare profile updates
        profile_updates = {
            "career_context": career_context,
            "learning_preferences": learning_preferences,
            "skill_profile": skill_profile,
        }
        
        # Generate welcome response
        welcome_message = recommendations.get(
            "welcomeMessage",
            f"Welcome! I've created your career profile. Your goal is to become a {career_context['target_role'][0] if career_context['target_role'] else 'Web3 professional'}. Let's get started!"
        )
        
        return {
            **state,
            "response": welcome_message,
            "profile_updates": profile_updates,
            "mode": "onboarding",
        }
    
    def career_guidance(self, state: StudentState) -> StudentState:
        """
        Provide ongoing career guidance to the student (conversational, not form-based).
        Uses existing profile data and conversation history for context-aware advice.
        """
        profile = state["user_profile"]
        completed = state["completed_courses"] or []
        summary = state.get("conversation_summary", "")
        
        # Build context from profile
        career_ctx = profile.get("career_context", {})
        target_roles = career_ctx.get("target_role", [])
        timeline = career_ctx.get("career_timeline", "flexible")
        current_status = career_ctx.get("current_status", "unknown")
        
        # Build conversation context
        context_parts = []
        if summary:
            context_parts.append(f"Previous conversation summary: {summary}")
        if target_roles:
            context_parts.append(f"Career goal: {', '.join(target_roles)}")
        if timeline:
            context_parts.append(f"Timeline: {timeline} months")
        if completed:
            context_parts.append(f"Completed courses: {', '.join([c.get('title', 'Unknown') for c in completed])}")
        
        context_str = "\n".join(context_parts) if context_parts else "New user, no prior context."

        system_prompt = f"""You are a career guidance AI for a Web3/AI learning platform.

User Context:
{context_str}

Current status: {current_status}

Your role:
- Provide personalized career advice aligned with their goals: {', '.join(target_roles) if target_roles else 'to be discovered'}
- Recommend specific learning tracks and courses on the platform
- Help them understand job market trends and requirements
- Guide them toward their {timeline}-month career timeline
- Be encouraging, specific, and actionable

If they ask about courses or tracks, recommend specific ones that align with their goals.
Be conversational and supportive."""
        
        # Build message history with summary
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (already trimmed by summarization if needed)
        messages.extend(state["conversation_history"])
        messages.append({"role": "user", "content": state["last_message"]})

        response = self.llm.invoke(messages)

        # Extract any new career insights from conversation (optional update)
        # Only extract if significant new information is mentioned
        extraction_prompt = f"""From this conversation, extract any NEW career-related information as JSON.
Only return data if the user provided NEW information not already in their profile.

User message: {state['last_message']}
Your response: {response.content}

Return ONLY valid JSON (or empty object {{}} if no new info):
{{
    "target_role": ["new role if mentioned"],
    "timeline": "updated timeline if mentioned",
    "motivation": "new motivation if mentioned"
}}"""

        extraction = self.llm.invoke([{"role": "user", "content": extraction_prompt}])

        try:
            new_career_data = json.loads(
                extraction.content.strip("```json").strip("```")
            )
        except Exception:
            new_career_data = {}

        # Only update if we got meaningful new data
        profile_updates = {}
        if new_career_data and any(new_career_data.values()):
            existing_career = profile.get("career_context", {})
            updated_career = {**existing_career}
            
            if new_career_data.get("target_role"):
                updated_career["target_role"] = new_career_data["target_role"]
            if new_career_data.get("timeline"):
                updated_career["career_timeline"] = new_career_data["timeline"]
            if new_career_data.get("motivation"):
                updated_career["primary_motivation"] = new_career_data["motivation"]
            
            profile_updates["career_context"] = updated_career

        return {
            **state,
            "response": response.content,
            "profile_updates": profile_updates,
        }

    def learning_assistance(self, state: StudentState) -> StudentState:
        """Help user while they're in a course, using frontend-supplied chapter context."""

        chapter_title = state.get("current_chapter_title")
        chapter_summary = state.get("current_chapter_summary")
        summary = state.get("conversation_summary", "")
        profile = state["user_profile"]
        
        # Get user's career context for personalized help
        career_ctx = profile.get("career_context", {})
        target_role = career_ctx.get("target_role", [])
        target_str = f" (aiming to become {target_role[0]})" if target_role else ""

        system_prompt = f"""You are a learning assistant helping a student in a Web3 course.

Current Chapter: {chapter_title or "Unknown"}
Chapter content summary:
{chapter_summary or "No summary available"}

User's skill level: {profile.get('career_context', {}).get('technical_level', 'Unknown')}{target_str}

{f'Previous conversation context: {summary}' if summary else ''}

Your role:
- Answer questions about the current chapter
- Explain concepts clearly with examples relevant to their career goals
- Provide examples and analogies
- Give hints for exercises (don't give full solutions)
- Connect concepts to their learning goals
- Encourage and motivate

Be patient and adaptive to their level."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(state["conversation_history"])
        messages.append({"role": "user", "content": state["last_message"]})

        response = self.llm.invoke(messages)

        # Track learning challenges based on difficulty signals
        if any(
            kw in state["last_message"].lower()
            for kw in ["don't understand", "confused", "stuck", "difficult"]
        ):
            challenges = state["user_profile"].get("learning_challenges", [])
            topic = chapter_title or "Current chapter"
            if topic not in challenges:
                challenges.append(topic)

            profile_updates = {"learning_challenges": challenges}
        else:
            profile_updates = {}

        return {**state, "response": response.content, "profile_updates": profile_updates}
    
    def progress_review(self, state: StudentState) -> StudentState:
        """Show user their learning progress."""

        completed = state["completed_courses"]
        current = state["current_course_id"]

        completed_lines = "\n".join(
            [f"✓ {c.get('title', 'Unknown course')}" for c in completed]
        )
        current_course_label = (
            f"Course ID {current}" if current is not None else "No active course"
        )

        progress_summary = f"""You've completed {len(completed)} courses:
{completed_lines}

Currently learning: {current_course_label}
"""

        system_prompt = f"""You are showing a student their learning progress.

{progress_summary}

Career goal: {state['user_profile'].get('career_context', {}).get('target_role', 'Not set')}

Provide:
- Celebration of achievements
- Progress toward career goal
- Encouragement
- Next recommended steps"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["last_message"]},
        ]

        response = self.llm.invoke(messages)

        return {**state, "response": response.content}

    def course_recommendation(self, state: StudentState) -> StudentState:
        """Recommend next courses based on goals and progress.

        This version does not call Web3 directly; it relies on the agent's
        general knowledge and the completed courses context.
        """

        career_goal = state["user_profile"].get("career_context", {}).get(
            "target_role", "Web3 developer"
        )

        completed_titles = [
            c.get("title", f"Course ID {c.get('course_id', '?')}")
            for c in state["completed_courses"]
        ]

        system_prompt = f"""You are a course recommendation assistant for a Web3 learning platform.

User wants to become: {career_goal}
They've completed: {completed_titles if completed_titles else "No courses yet"}

Based on this, suggest the top 3 next course topics or learning modules they should take,
and explain briefly why each is important for their goal. You don't know the exact
course catalog; focus on topics and learning objectives."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["last_message"]},
        ]

        response = self.llm.invoke(messages)

        return {**state, "response": response.content}
    
    def general_conversation(self, state: StudentState) -> StudentState:
        """General helpful conversation."""

        system_prompt = f"""You are a friendly learning companion for a Web3 education platform.

User's goal: {state['user_profile'].get('career_context', {}).get('target_role', 'learning Web3')}

Be helpful, encouraging, and guide them toward their learning goals."""

        messages = [
            {"role": "system", "content": system_prompt},
            *state["conversation_history"],
            {"role": "user", "content": state["last_message"]},
        ]

        response = self.llm.invoke(messages)

        return {**state, "response": response.content}

    def update_user_profile_node(self, state: StudentState) -> StudentState:
        """Save any profile updates to DB and log the conversation."""

        if state.get("profile_updates"):
            update_user_profile(
                self.db, state["wallet_address"], state["profile_updates"]
            )

        # Save conversation
        save_conversation(
            self.db, state["wallet_address"], "user", state["last_message"]
        )
        save_conversation(
            self.db, state["wallet_address"], "assistant", state["response"]
        )

        return state

    def invoke(self, initial_state: dict) -> dict:
        """Execute the compiled agent graph."""
        return self.graph.invoke(initial_state)
