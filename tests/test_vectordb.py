import pytest
from conftest import FakeReranker
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

import vectordb


@pytest.fixture
def index():
    documents = [Document(page_content=f"passage {i}", metadata={"source": "ports.pdf", "page": i}) for i in range(6)]
    return FAISS.from_documents(documents, DeterministicFakeEmbedding(size=16))


def test_without_reranker_returns_top_k_vector_hits(index):
    docs = vectordb.retrieve_documents(index, "passage 3", search_method="similarity", vector_search_top_k=4)
    assert len(docs) == 4
    assert docs[0].page_content == "passage 3"  # identical text has an identical fake embedding


def test_reranker_narrows_to_top_n_and_keeps_source_and_page(index):
    docs = vectordb.retrieve_documents(
        index,
        "passage 3",
        reranker=FakeReranker(),
        search_method="similarity",
        vector_search_top_k=4,
        vector_rerank_top_n=2,
    )
    assert len(docs) == 2
    for doc in docs:
        assert doc.metadata["source"] == "ports.pdf"
        assert doc.page_content == f"passage {doc.metadata['page']}"  # metadata belongs to the right passage
        assert isinstance(doc.metadata["relevance_score"], float)
    assert docs[0].metadata["relevance_score"] > docs[1].metadata["relevance_score"]


def test_rerank_top_n_larger_than_search_top_k_is_an_error(index):
    with pytest.raises(ValueError, match="top k"):
        vectordb.retrieve_documents(index, "q", vector_search_top_k=2, vector_rerank_top_n=3)


def test_unsupported_document_type_is_rejected():
    with pytest.raises(ValueError):
        vectordb.load_single_document("notes.docx")


def test_local_embedding_export_is_preferred_over_the_hub_id(tmp_path, monkeypatch):
    export = tmp_path / "BAAI" / "bge-small-en-v1.5"
    export.mkdir(parents=True)
    (export / "openvino_model.xml").write_text("<net/>")
    monkeypatch.setattr(vectordb, "embedding_models_dir", tmp_path)
    loaded = {}
    monkeypatch.setattr(vectordb, "OpenVINOBgeEmbeddings", lambda **kwargs: loaded.update(kwargs) or "embedder")
    assert vectordb.load_ov_embedding_model("BAAI/bge-small-en-v1.5") == "embedder"
    assert loaded["model_name_or_path"] == str(export)


def test_embedding_falls_back_to_the_hub_id_without_a_local_export(tmp_path, monkeypatch):
    monkeypatch.setattr(vectordb, "embedding_models_dir", tmp_path)
    loaded = {}
    monkeypatch.setattr(vectordb, "OpenVINOBgeEmbeddings", lambda **kwargs: loaded.update(kwargs) or "embedder")
    vectordb.load_ov_embedding_model("BAAI/bge-small-en-v1.5")
    assert loaded["model_name_or_path"] == "BAAI/bge-small-en-v1.5"
