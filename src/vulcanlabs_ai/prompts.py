# --- Session summary (structured output) ---
SESSION_SUMMARY_SYSTEM = (
    "You are a conversation summarizer. Output a structured summary with topics, key_facts, user_goals, and summary_text."
)
SUMMARY_EXTEND_INSTRUCTION = (
    "This is the existing session summary:\n{summary_text}\n\n"
    "Extend the summary by taking into account the new messages above."
)
SUMMARY_CREATE_INSTRUCTION = (
    "Create a structured summary of the conversation above between AI and the user. "
    "The summary must capture all relevant topics, facts, and user goals."
)

# --- Query ambiguity (structured output) ---
QUERY_AMBIGUITY_SYSTEM = (
    "Analyze whether the user's query is ambiguous or unclear. "
    "Output is_ambiguous (bool), and optionally ambiguity_reason and rewritten_query."
)

# --- Clarifying questions (structured output) ---
CLARIFYING_QUESTIONS_SYSTEM = (
    "Determine if the user's query needs clarification. "
    "If so, set needs_clarification=true and provide 1-3 clarifying questions; "
    "otherwise set needs_clarification=false."
)

# --- Augment context ---
SESSION_SUMMARY_PREFIX = "Session summary: {summary}"

# --- Question node (clarifying questions reply) ---
CLARIFYING_INTRODUCTION = "I need a bit more information before proceeding:\n"
