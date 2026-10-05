"""Offline evaluation harness: runs the RAG graph over evals/dataset.py and
scores each answer three ways:

1. grounded  — the graph's own verify node verdict (LLM-as-judge against the
   retrieved documents)
2. relevant  — an LLM judge checking the answer addresses the question
3. keywords  — deterministic check that an expected fact substring appears

Requires an LLM key (ANTHROPIC_API_KEY or OPENAI_API_KEY) in .env — this is
a manual harness, not a pytest test. Usage:

    python -m evals.run_eval            # run all questions
    python -m evals.run_eval --verbose  # also print each answer + judge verdict
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.prompts import ChatPromptTemplate

from app.agent_main import get_agent_llm
from app.main import get_or_build_vector_store
from evals.dataset import QUESTIONS
from graph.employee_graph import build_employee_graph
from graph.nodes import _check_grounded
from rag.prompts import format_docs
from rag.retriever import get_retriever

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator. Decide whether the answer addresses the "
            "question asked — factually correct or not is a different judge's "
            "problem; here you only check the answer is on-topic and responsive. "
            "If the question asks something the answer admits it cannot answer, "
            "that counts as responsive. Reply with exactly 'RELEVANT' or "
            "'IRRELEVANT: <one-line reason>'.",
        ),
        ("human", "Question: {question}\n\nAnswer: {answer}"),
    ]
)


def _check_relevant(llm, question: str, answer: str) -> bool:
    verdict = llm.invoke(
        RELEVANCE_PROMPT.invoke({"question": question, "answer": answer})
    ).content.strip()
    return verdict.upper().startswith("RELEVANT") and not verdict.upper().startswith(
        "IRRELEVANT"
    )


def _keywords_hit(answer: str, expected_any: list[str]) -> bool:
    lowered = answer.lower()
    return any(expected.lower() in lowered for expected in expected_any)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    documents_dir = os.getenv("DOCUMENTS_DIR", "documents")
    persist_dir = os.getenv("VECTOR_STORE_DIR", ".vector_store")
    store = get_or_build_vector_store(documents_dir, persist_dir)
    retriever = get_retriever(store)
    llm = get_agent_llm()
    graph = build_employee_graph(llm, retriever)

    results = []
    for case in QUESTIONS:
        question = case["question"]
        state = graph.invoke(
            {"question": question, "chat_history": [], "documents": [], "answer": ""}
        )
        answer = state["answer"]
        evidence = format_docs(state["documents"])

        # Re-judge externally rather than trusting the graph's own flag, so
        # the harness stays an independent check even if the graph changes.
        # Abstention cases are skipped: "the info isn't available" can't be
        # supported by the sources by definition — keywords+relevance judge it.
        if case.get("abstain"):
            grounded, verdict = None, "n/a (abstention expected)"
        else:
            grounded, verdict = _check_grounded(llm, evidence, answer)
        relevant = _check_relevant(llm, question, answer)
        keywords = _keywords_hit(answer, case.get("expected_any", []))
        results.append((case, answer, grounded, verdict, relevant, keywords))

        marks = " ".join(
            [
                f"grounded={'n/a' if grounded is None else ('Y' if grounded else 'N')}",
                f"relevant={'Y' if relevant else 'N'}",
                f"keywords={'Y' if keywords else 'N'}",
            ]
        )
        ok = (grounded is not False) and relevant and keywords
        print(f"[{'PASS' if ok else 'FAIL'}] {question}")
        print(f"      {marks}")
        if args.verbose:
            print(f"      answer : {answer}")
            print(f"      verdict: {verdict}")

    total = len(results)
    passed = sum(1 for r in results if r[2] is not False and r[4] and r[5])
    grounded_count = sum(1 for r in results if r[2] is True)
    judged_count = sum(1 for r in results if r[2] is not None)
    relevant_count = sum(1 for r in results if r[4])
    keyword_count = sum(1 for r in results if r[5])
    print(
        f"\n{passed}/{total} passed | grounded {grounded_count}/{judged_count} judged | "
        f"relevant {relevant_count}/{total} | keywords {keyword_count}/{total}"
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
