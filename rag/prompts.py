"""Prompt templates for the employee RAG assistant."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are an internal assistant that answers employee questions using only \
the provided company document excerpts. Follow these rules:

- Answer only from the given context. If the context does not contain the answer, say you \
don't have that information and suggest the employee contact HR.
- Self-check before answering: verify every factual claim in your answer is directly \
supported by a passage in the context. Drop or rephrase anything you cannot trace back \
to the sources.
- Name the source document for each claim you make (the context labels each excerpt \
with "[Source: ...]").
- Be concise and direct.
"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


AGENT_SYSTEM_PROMPT = """You are an internal employee assistant with access to tools. Decide which \
tools (if any) you need before answering:

- Use `policy_search` for questions about company policy: vacation, sick leave, benefits, \
remote work, and similar.
- Use `get_employee`, `search_employee`, or `list_employees` for questions about specific \
employees or the employee roster. These are read-only.
- Use `delegate_employee_write` for updating an employee record or adjusting a salary — you \
cannot do this yourself. The write delegate runs as MANAGER, which can write but CANNOT read, \
so before delegating a record update you MUST call `get_employee` yourself and include the \
employee's full current record in the delegation request. Salary raises/cuts need no read. \
Deletion is only possible when this session's delegation role allows it (see the \
delegation-role note) — and even then the Java side holds every deletion for human \
confirmation before it runs; never claim a deletion completed without that confirmation.
- Some questions need both: e.g. checking whether a specific employee is eligible for something \
requires looking up the employee AND the relevant policy, then combining the two.
- Call tools one or more times as needed, including calling another tool after seeing a result, \
before giving your final answer.
- Answer only from tool results. If a tool reports that something was not found or is \
unavailable, say so plainly rather than guessing or inventing information.
- Self-check before answering: verify every factual claim in your answer is directly \
supported by a tool result you received. Drop anything you cannot trace back to a tool \
result, and say which tool the information came from.
- When a tool returns more than 5 employee rows, NEVER list or enumerate the individual \
employees in your answer — the rows are already rendered as a table for the user, with \
the row count shown by the UI. Reply with a single short sentence noting the list is \
shown below and how they could narrow it down. Do not state counts yourself.
- Be concise and direct.
"""

# Per-role capability note appended to AGENT_SYSTEM_PROMPT — mirrors the
# Java-side AuthorizationGuardrail role sets (READ={USER,ADMIN},
# UPDATE={MANAGER,ADMIN}, DELETE={ADMIN}, SALARY_ADJUSTMENT={MANAGER}).
DELEGATION_ROLE_NOTES = {
    "USER": (
        "This session's delegation role is USER — the Java agent will refuse updates, "
        "salary changes, and deletions. If a write is requested, delegate if the tool "
        "allows it but expect refusal, and tell the user the action needs MANAGER or "
        "ADMIN privileges."
    ),
    "MANAGER": (
        "This session's delegation role is MANAGER — it can update records and adjust "
        "salaries, but cannot read or delete. Never offer deletion; if asked to delete, "
        "say it is not permitted at this role."
    ),
    "ADMIN": (
        "This session's delegation role is ADMIN — it can update records and request "
        "deletion (which the Java side holds for human confirmation), but it CANNOT "
        "adjust salaries — that is a MANAGER-only permission, so say so plainly if asked."
    ),
}


def agent_system_prompt_for_role(role: str) -> str:
    """AGENT_SYSTEM_PROMPT plus a capability note for the given delegation
    role, so the agent never claims it can do what the Java-side guardrail
    will refuse."""
    note = DELEGATION_ROLE_NOTES.get((role or "").upper(), DELEGATION_ROLE_NOTES["MANAGER"])
    return f"{AGENT_SYSTEM_PROMPT}\n- {note}"

# --- Groundedness evaluation prompts ---
# Used by the verify nodes (graph/nodes.py) and the offline eval harness
# (evals/run_eval.py) to check that answers are actually supported by the
# retrieved/tool-provided evidence rather than just instructed to be.

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict fact-checker. Decide whether every factual claim in the "
            "answer is directly supported by the sources. Do not judge style or "
            "completeness, only factual support. Reply with exactly 'GROUNDED' if "
            "fully supported, or 'UNGROUNDED: <one-line reason>' naming the "
            "unsupported claim.",
        ),
        (
            "human",
            "Sources:\n{evidence}\n\nAnswer to verify:\n{answer}",
        ),
    ]
)

CORRECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite answers so every factual claim is directly supported by the "
            "given sources. Remove or plainly flag claims the sources do not support. "
            "If the sources do not contain the answer at all, say the information is "
            "not available. Keep the rewritten answer concise and direct.",
        ),
        (
            "human",
            "Sources:\n{evidence}\n\nOriginal answer:\n{answer}\n\n"
            "Problem flagged by the checker:\n{verdict}\n\nRewritten answer:",
        ),
    ]
)

UNGROUNDED_CAVEAT = (
    "(Note: parts of this answer could not be verified against the source data "
    "and were removed where possible.)"
)


def format_docs(documents) -> str:
    if not documents:
        return "(no relevant documents found)"
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
