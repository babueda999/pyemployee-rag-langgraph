"""Combines the retriever, prompt, and LLM into a single answerable chain."""

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from rag.prompts import ANSWER_PROMPT, format_docs
from rag.retriever import Retriever


@dataclass
class RagResult:
    answer: str
    source_documents: list[Document] = field(default_factory=list)


class RagChain:
    def __init__(self, llm: BaseChatModel, retriever: Retriever):
        self.llm = llm
        self.retriever = retriever

    def invoke(self, question: str) -> RagResult:
        documents = self.retriever.retrieve(question)
        prompt_value = ANSWER_PROMPT.invoke(
            {"context": format_docs(documents), "question": question}
        )
        response = self.llm.invoke(prompt_value)
        return RagResult(answer=response.content, source_documents=documents)


def build_rag_chain(llm: BaseChatModel, retriever: Retriever) -> RagChain:
    return RagChain(llm, retriever)
