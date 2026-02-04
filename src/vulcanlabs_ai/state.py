from langchain_core.messages.base import BaseMessage
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import Optional


###################
# Structured Outputs
###################
class SessionSummary(BaseModel):
    topics: list[str] | None = Field(default=None, description="Main discussion topics")
    key_facts: list[str] | None = Field(default=None, description="Important facts mentioned")
    user_goals: list[str] | None = Field(default=None, description="User's stated or inferred goals")
    summary_text: str = Field(default="", description="Concise summary of the session")

class QueryAmbiguityResult(BaseModel):
    is_ambiguous: bool = Field(
        description="Whether the user's query is ambiguous or lacks sufficient clarity."
    )

    ambiguity_reason: Optional[str] = Field(
        default=None,
        description="Short explanation of why the query is ambiguous."
    )

    rewritten_query: Optional[str] = Field(
        default=None,
        description=(
            "A rewritten version of the original query that resolves ambiguity by choosing a reasonable interpretation, without asking the user a question."
        )
    )
    
class ClarifyingQuestionResult(BaseModel):
    needs_clarification: bool = Field(
        description="Whether the system needs clarification from the user"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Short explanation of why clarification is needed"
    )
    questions: list[str] | None = Field(
        default=None,
        description="1-3 clarifying questions to ask the user",
    )



###################
# State Definitions
###################
class State(MessagesState):
    summary: SessionSummary | None
    query_ambiguity: QueryAmbiguityResult | None
    clarifying_question: ClarifyingQuestionResult | None
    augmented_context: list[BaseMessage] | None
