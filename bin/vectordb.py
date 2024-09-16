from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from config import vectorstore_path, rag_prompt_template, DEFAULT_RAG_PROMPT, pdf_path
import gradio as gr
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader
import logging

logging.basicConfig(level=logging.INFO)

TEXT_SPLITERS = {
    "Character": CharacterTextSplitter,
    "RecursiveCharacter": RecursiveCharacterTextSplitter,
    "Markdown": MarkdownTextSplitter,
}

LOADERS = {".pdf": (PyPDFLoader, {})}  # can be also used PyPDFium2  for pdf loading seems more faster



def load_hf_embedding_model():
    """
    Load the huggingface model
    Returns: model
    """
    model_name = "BAAI/bge-small-en-v1.5" #sentence-transformers/all-mpnet-base-v2" #"BAAI/bge-small-en" # this is download the model from huggingface i have downloaded the model in openvino format bu it does not work
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'mean_pooling': False,'normalize_embeddings': True, "batch_size": 4}
    hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
    
    return hf

def load_reranker_model():
    """
    Load the reranker model
    Returns: model
    """
    try:
        rerank_model_name = "../reranker/bge-reranker-v2-m3"
        rerank_model_kwargs = {"device": "CPU"}
        rerank_top_n = 2

        reranker = OpenVINOReranker(
        model_name_or_path=rerank_model_name,
        model_kwargs=rerank_model_kwargs,
        top_n=rerank_top_n,
        )
        logging.info("Loading reranker")
        return reranker
    except Exception as e:
        raise ValueError("Error loading reranker{}".format(e))

def load_single_document(file_path: str) -> List[Document]:
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


def create_vectordb(docs, spliter_name, chunk_size, chunk_overlap ):
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

    text_splitter = TEXT_SPLITERS[spliter_name](chunk_size=chunk_size, chunk_overlap=chunk_overlap,)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, load_hf_embedding_model())
    db.save_local(vectorstore_path)

    return "Vectorstore created at {}".format(vectorstore_path)

def create_rag_chain(db, llm, vector_search_top_k, vector_rerank_top_n, reranker, search_method, score_threshold):
    """
    Create a RAG chain from a vectorstore

    Params:
      db: vectorstore
      llm: language model
      vector_search_top_k: top k for search
      vector_rerank_top_n: top n for rerank
      run_rerank: run rerank
      search_method: search method
      score_threshold: score threshold
    Returns:
      RAG chain
    """ 
    if vector_rerank_top_n > vector_search_top_k:
        raise ValueError("Search top k must >= Rerank top n")

    if search_method == "similarity_score_threshold":
        search_kwargs = {"k": vector_search_top_k, "score_threshold": score_threshold}
    else:
        search_kwargs = {"k": vector_search_top_k}
    retriever = db.as_retriever(search_kwargs=search_kwargs, search_type=search_method)
    if reranker:
        reranker.top_n = vector_rerank_top_n
        retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=retriever)
    prompt = PromptTemplate(input_variables=["DEFAULT_RAG_PROMPT", "context", "question"], template=rag_prompt_template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

#add the update retriever method


create_vectordb([pdf_path], "RecursiveCharacter", 400, 50)