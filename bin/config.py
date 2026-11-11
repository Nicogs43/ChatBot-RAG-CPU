from pathlib import Path

# All paths are resolved from the repository root, so the scripts work from any directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INFERENCE_TIME_DIR = DATA_DIR / "inference_time"
QUESTIONS_PATH = DATA_DIR / "questions_ground_truths_25.csv"

vectorstore_path = str(ROOT / "vectorstore_ov_400_50")
pdf_path = str(ROOT / "samples" / "Ports_chapters_10_to_20.pdf")

# The embedding model used to query FAISS must match the one used to build it.
embedding_model_id = "BAAI/bge-small-en-v1.5"
embedding_models_dir = ROOT / "models"  # download_embedding_model.py exports here
reranker_path = str(ROOT / "reranker" / "bge-reranker-v2-m3_quantized")

DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks about maritime domain. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

QWEN_DEFAULT_RAG_PROMPT = """\
You are a knowledgeable assistant specializing in answering questions about ports, sailing, and maritime activities along the Italian peninsula. Use the most relevant pieces of the retrieved context below to support your answer. If you don't know the answer or the context is insufficient, state that you are unsure. Use a maximum of three sentences, and keep your answer concise, accurate, and where applicable, use maritime terminology."""

# The user turn. The system prompt and the model-specific chat markers are added
# by the tokenizer's own chat template (see chatbot.RagEngine.build_prompt).
USER_PROMPT_TEMPLATE = "Question: {input}\nContext: {context}"

ov_config = {
    "KV_CACHE_PRECISION": "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}

generation_kwargs = dict(
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
)

retrieval_config = dict(
    search_method="similarity_score_threshold",
    vector_search_top_k=5,
    vector_rerank_top_n=2,
    score_threshold=0.6,
)


def _model(relative_path: str, system_prompt: str) -> dict:
    return {"model_path": str(ROOT / "model" / relative_path), "system_prompt": system_prompt}


# Short model key -> local OpenVINO IR folder and system prompt. The key also
# prefixes the output CSV filenames in data/. To add a model, export it with
# download_model.py and add an entry here.
model_dict = {
    "gemma": _model("google/gemma-2-2b-it", DEFAULT_RAG_PROMPT),
    "phi_int4": _model("microsoft/Phi-3.5-mini-instruct/int4", DEFAULT_RAG_PROMPT),
    "phi_int8": _model("microsoft/Phi-3.5-mini-instruct/int8", DEFAULT_RAG_PROMPT),
    "qwen_0.5b": _model("Qwen/Qwen2.5-0.5B-Instruct", QWEN_DEFAULT_RAG_PROMPT),
    "qwen_1.5b": _model("Qwen/Qwen2.5-1.5B-Instruct", QWEN_DEFAULT_RAG_PROMPT),
    "qwen_3b": _model("Qwen/Qwen2.5-3B-Instruct/int8", QWEN_DEFAULT_RAG_PROMPT),
}

default_model = "qwen_0.5b"


def available_models() -> list:
    """Keys of model_dict whose exported model is present on disk."""
    return [key for key, details in model_dict.items()
            if (Path(details["model_path"]) / "openvino_model.xml").exists()]


def model_label(model_key: str) -> str:
    """Readable model name for the UI, e.g. 'Phi-3.5-mini-instruct (int4)'."""
    parts = Path(model_dict[model_key]["model_path"]).relative_to(ROOT / "model").parts[1:]
    name, variant = parts[0], parts[1:]
    return f"{name} ({variant[0]})" if variant else name
