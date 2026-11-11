import logging
import time
from pathlib import Path
from typing import Any

from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import embedding_model_id, embedding_models_dir, reranker_path, vectorstore_path

logger = logging.getLogger(__name__)

TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
}

LOADERS = {".pdf": (PyPDFLoader, {})}  # can be also used PyPDFium2  for pdf loading seems more faster


def load_ov_embedding_model(model_name_or_path: str = embedding_model_id) -> OpenVINOBgeEmbeddings:
    """
    Args:
        model_name_or_path (str): HF id or local folder of the embedding model.
    Returns:
        OpenVINOBgeEmbeddings: The loaded model.

    If the model was already exported with download_embedding_model.py, the local
    copy in models/ is used. Otherwise it is downloaded and converted to OpenVINO
    at every start, which is much slower.
    """
    local_copy = embedding_models_dir / model_name_or_path
    if (local_copy / "openvino_model.xml").exists():
        model_name_or_path = str(local_copy)
    elif not Path(model_name_or_path).exists():
        logger.warning(
            "No local OpenVINO export of %s: converting it now. Run download_embedding_model.py once to skip this.",
            model_name_or_path,
        )
    try:
        logger.info("Loading openvino embedding model from %s", model_name_or_path)
        return OpenVINOBgeEmbeddings(  # using this class has the right value for query_instruction for bge models
            model_name_or_path=model_name_or_path,
            model_kwargs={"device": "CPU"},
            encode_kwargs={"mean_pooling": False, "normalize_embeddings": True, "batch_size": 4},
            show_progress=True,
        )
    except Exception as e:
        raise ValueError(f"Error loading openvino embedding model: {e}") from e


def load_reranker_model(rerank_model_name: str = reranker_path, top_n: int = 2) -> OpenVINOReranker:
    """
    Args:
        rerank_model_name (str): local folder (or HF id) of the reranker model.
        top_n (int): number of documents kept after reranking.
    Returns:
        OpenVINOReranker: the loaded model.
    """
    try:
        logger.info("Loading reranker from %s", rerank_model_name)
        return OpenVINOReranker(
            model_name_or_path=rerank_model_name,
            model_kwargs={"device": "CPU"},
            top_n=top_n,
        )
    except Exception as e:
        raise ValueError(f"Error loading reranker: {e}") from e


def load_single_document(file_path: str) -> list[Document]:
    """
    helper for loading a single document
    Params:
      file_path: document path
    Returns:
      documents loaded
    """
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADERS:
        loader_class, loader_args = LOADERS[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"File does not exist '{ext}'")


def create_vectordb(
    docs, spliter_name, chunk_size, chunk_overlap, embedding_model, vector_store_name=vectorstore_path
) -> str:
    """
    Create a vectorstore from a list of documents

    Params:
      docs: list of documents
      spliter_name: text splitter name
      chunk_size: chunk size
      chunk_overlap: chunk overlap
      vectorstore_path: path to save vectorstore
    Returns:
      vectorstore
    """
    documents = []
    for doc in docs:
        if type(doc) is not str:
            doc = doc.name
        documents.extend(load_single_document(doc))

    text_splitter = TEXT_SPLITERS[spliter_name](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = text_splitter.split_documents(documents)
    start_time = time.time()
    vector_index = FAISS.from_documents(texts, load_ov_embedding_model(embedding_model))
    print("--- %s seconds ---" % (time.time() - start_time))
    vector_index.save_local(vector_store_name)

    return f"Vectorstore created at {vector_store_name}"


def retrieve_documents(
    vector_index: FAISS,
    question: str,
    reranker: OpenVINOReranker | None = None,
    search_method: str = "similarity_score_threshold",
    vector_search_top_k: int = 5,
    vector_rerank_top_n: int = 2,
    score_threshold: float = 0.6,
) -> list[Document]:
    """
    Retrieve the documents used as context for a question.

    Params:
        vector_index: vectorstore
        question: the user question
        reranker: reranker instance (optional)
        search_method: search method
        vector_search_top_k: top k for search
        vector_rerank_top_n: top n kept after rerank
        score_threshold: score threshold (only for similarity_score_threshold)
    Returns:
      the vector search results, narrowed down by the reranker if there is one
    """
    if vector_rerank_top_n > vector_search_top_k:
        raise ValueError("Search top k must >= Rerank top n")

    search_kwargs: dict[str, Any] = {"k": vector_search_top_k}
    if search_method == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold
    retriever = vector_index.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
    docs = retriever.invoke(question)
    if reranker and docs:
        reranker.top_n = vector_rerank_top_n
        reranked = []
        for hit in reranker.compress_documents(docs, question):
            # The reranker returns bare copies: metadata is only {"id": position in docs, "relevance_score"}.
            # Put the original metadata (source, page) back.
            original = docs[hit.metadata["id"]] if "id" in hit.metadata else hit
            score = float(hit.metadata.get("relevance_score", 0.0))
            reranked.append(
                Document(page_content=hit.page_content, metadata={**original.metadata, "relevance_score": score})
            )
        docs = reranked
    return docs
