"""Shared fakes: the tests never load a real model, index or reranker."""

import pytest
import torch
from jinja2.exceptions import TemplateError
from langchain_core.documents import Document

import chatbot


class FakeTokenizer:
    """Mimics the parts of a HF tokenizer that RagEngine uses."""

    def __init__(self, supports_system=True):
        self.supports_system = supports_system
        self.seen_messages = None

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        if not self.supports_system and any(m["role"] == "system" for m in messages):
            raise TemplateError("System role not supported")
        self.seen_messages = messages
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages) + "<assistant>"
        return {"input_ids": [[1, 2, 3]]} if tokenize else text

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


class FakeModel:
    """Streams `words` one at a time, checking the stopping criteria between words like a real model."""

    def __init__(self, words=("Channel ", "12 ", "and ", "16."), error=None):
        self.words = words
        self.error = error
        self.calls = 0
        self.words_sent = 0
        self.kwargs = None

    def generate(self, streamer=None, stopping_criteria=None, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.error:
            raise self.error
        for word in self.words:
            streamer.on_finalized_text(word)
            self.words_sent += 1
            if stopping_criteria and bool(stopping_criteria(torch.zeros((1, 1), dtype=torch.long), None).all()):
                break
        streamer.end()


class FakeReranker:
    """Like OpenVINOReranker: returns bare copies whose metadata is only id + relevance_score, best first."""

    top_n = 2

    def compress_documents(self, documents, query):
        ranked = list(enumerate(documents))[::-1][: self.top_n]  # pretend the last hits are the best
        return [
            Document(page_content=doc.page_content, metadata={"id": i, "relevance_score": 0.9 - 0.1 * rank})
            for rank, (i, doc) in enumerate(ranked)
        ]


@pytest.fixture
def docs():
    return [
        Document(page_content="Imperia: VHF channels 12 and 16.", metadata={"source": "ports.pdf", "page": 2}),
        Document(page_content="Genoa: winds from the north.", metadata={"source": "ports.pdf", "page": 7}),
    ]


@pytest.fixture
def make_engine(monkeypatch, docs):
    """Build a RagEngine around fakes. Returns (engine, model, tokenizer)."""

    def _make(model=None, tokenizer=None, retrieved=None):
        model = model or FakeModel()
        tokenizer = tokenizer or FakeTokenizer()
        monkeypatch.setattr(chatbot, "load_openvino_model", lambda path: (model, tokenizer))
        monkeypatch.setattr(chatbot, "retrieve_documents", lambda *a, **k: docs if retrieved is None else retrieved)
        engine = chatbot.RagEngine("qwen_0.5b", vectorstore=object(), reranker=None)
        return engine, model, tokenizer

    return _make
