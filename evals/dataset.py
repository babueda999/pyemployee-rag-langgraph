"""Fixed question set for the offline eval (evals/run_eval.py).

Each case:
- question: what to ask the RAG graph
- expected_any: at least one of these substrings must appear in the answer
  (case-insensitive) — a deterministic keyword check alongside the LLM judges
- abstain: True when the correct behavior is to admit the info isn't in the
  documents instead of answering

Facts come from documents/sample_policy.txt. Keep them in sync if the
document changes.
"""

QUESTIONS = [
    {
        "question": "How many days of PTO do full-time employees accrue per year?",
        "expected_any": ["15"],
    },
    {
        "question": "How much unused PTO can I carry into next year?",
        "expected_any": ["5 unused", "5 day", "maximum of 5", "up to 5"],
    },
    {
        "question": "When can a new employee start using accrued PTO?",
        "expected_any": ["90 day", "90-day"],
    },
    {
        "question": "How far in advance should I request planned time off?",
        "expected_any": ["5 business day", "five business day"],
    },
    {
        "question": "What happens to unused PTO above the carryover limit?",
        "expected_any": ["forfeit"],
    },
    {
        "question": "What is the company's parental leave policy?",
        "expected_any": [
            "don't have that information",
            "do not have that information",
            "not in the context",
            "no information",
            "not available",
            "contact hr",
        ],
        "abstain": True,
    },
]
