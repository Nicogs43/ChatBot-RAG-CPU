import logging
import threading
from typing import Iterator, List, Optional

import torch
from jinja2.exceptions import TemplateError
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from config import (
    USER_PROMPT_TEMPLATE,
    generation_kwargs,
    model_dict,
    ov_config,
    retrieval_config,
    vectorstore_path,
)
from vectordb import load_ov_embedding_model, load_reranker_model, retrieve_documents

logger = logging.getLogger(__name__)


def load_openvino_model(model_path: str, ov_config: dict = ov_config):
    """
    Load an OpenVINO IR model and its tokenizer on CPU.
    Args:
        model_path: folder with the exported OpenVINO model and its tokenizer
        ov_config: OpenVINO runtime configuration
    Returns: (model, tokenizer)
    """
    logger.info("Loading openvino model from %s", model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = OVModelForCausalLM.from_pretrained(model_path, device="CPU", ov_config=ov_config)
    except Exception as e:
        raise ValueError(f"Error loading openvino model: {e}") from e
    return model, tokenizer


def load_retrieval_resources(use_reranker: bool = True):
    """Load what every model shares: the FAISS index (with its embedding model) and the reranker."""
    logger.info("Loading vectorstore from %s", vectorstore_path)
    try:
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=load_ov_embedding_model(),
            allow_dangerous_deserialization=True,  # the index is a local pickle
        )
    except Exception as e:
        raise ValueError(f"Error loading vectorstore: {e}") from e
    reranker = load_reranker_model() if use_reranker else None
    return vectorstore, reranker


class _StopFlag(StoppingCriteria):
    """Ends generation after the current token once `event` is set."""

    def __init__(self):
        self.event = threading.Event()

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(), dtype=torch.bool)


class RagEngine:
    """Retrieval + generation for one model of config.model_dict.

    The vectorstore and reranker are passed in so that several engines (one per
    model, as in the evaluation scripts) can share them.
    """

    def __init__(self, model_key: str, vectorstore: FAISS, reranker=None):
        if model_key not in model_dict:
            raise KeyError(f"Unknown model '{model_key}'. Available: {', '.join(model_dict)}")
        self.model_key = model_key
        self.system_prompt = model_dict[model_key]["system_prompt"]
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.model, self.tokenizer = load_openvino_model(model_dict[model_key]["model_path"])
        self._stop = _StopFlag()
        self._lock = threading.Lock()  # one generation at a time per model

    def retrieve(self, question: str) -> List[Document]:
        return retrieve_documents(self.vectorstore, question, reranker=self.reranker, **retrieval_config)

    def build_messages(self, question: str, docs: List[Document]) -> List[dict]:
        context = "\n\n".join(doc.page_content for doc in docs)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(input=question, context=context)},
        ]

    def _apply_chat_template(self, messages: List[dict], **kwargs):
        """Render the messages with the model's own chat template."""
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
        except TemplateError:
            # Some templates (e.g. Gemma) have no system role: put the system prompt in the user turn.
            system, user = messages
            merged = [{"role": "user", "content": f"{system['content']}\n\n{user['content']}"}]
            return self.tokenizer.apply_chat_template(merged, add_generation_prompt=True, **kwargs)

    def build_prompt(self, question: str, docs: List[Document]) -> str:
        """The exact prompt text the model receives (useful for inspection)."""
        return self._apply_chat_template(self.build_messages(question, docs), tokenize=False)

    def stream(self, question: str, docs: Optional[List[Document]] = None) -> Iterator[str]:
        """Yield the answer piece by piece as it is generated.

        There is no timeout: on a slow CPU a large model can take more than a
        minute before the first token. Closing the generator early, or calling
        cancel() from another thread, stops the generation.
        """
        if docs is None:
            docs = self.retrieve(question)
        inputs = self._apply_chat_template(
            self.build_messages(question, docs), tokenize=True, return_dict=True, return_tensors="pt"
        )
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        errors = []

        def generate():
            try:
                self.model.generate(
                    **inputs,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([self._stop]),
                    **generation_kwargs,
                )
            except Exception as e:  # surface the error in the caller's thread
                errors.append(e)
                streamer.end()

        with self._lock:
            self._stop.event.clear()
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            try:
                for piece in streamer:
                    if piece:
                        yield piece
            finally:
                self._stop.event.set()  # no-op if finished; stops the model if the caller gave up
                thread.join()
        if errors:
            raise errors[0]

    def invoke(self, question: str) -> dict:
        """Answer a question. Returns {"input", "context" (retrieved documents), "answer"}."""
        docs = self.retrieve(question)
        answer = "".join(self.stream(question, docs))
        return {"input": question, "context": docs, "answer": answer.strip()}

    def cancel(self) -> None:
        """Stop the generation in progress, if any (it ends after the current token)."""
        self._stop.event.set()
