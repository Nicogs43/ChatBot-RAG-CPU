from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from config import vectorstore_path, phi_rag_prompt_template, qwen_rag_prompt_template, pdf_path
import time
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


#provare ad usare huggingfacebgeembeddings e poi provare ad inserire con bge-small-en-v1.5 il parametro query_instruction
def load_hf_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceBgeEmbeddings:
    """
    Args:
        model_name (str): The name of the model to load.
    Returns: 
        HuggingFaceBgeEmbeddings: The loaded model.
    """
    try:
        logging.info("Loading huggingface embedding model")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'mean_pooling': False,'normalize_embeddings': True, "batch_size": 4}
        hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        #query_instruction="", #this is the query instruction because bge-m3 don't need a query instruction
        #using a different model from bge-m3 you can use the query_instruction parameter or leave it as setted by default
        )
        return hf
    except Exception as e:
        raise ValueError("Error loading huggingface embedding model{}".format(e))
    
def load_ov_embedding_model(model_name_or_path: str = "BAAI/bge-large-en-v1.5") -> OpenVINOBgeEmbeddings:
    """
    Args:
        model_name_or_path (str): The name of the model to load.
    Returns: 
        OpenVINOBgeEmbeddings: The loaded model.
    """
    try:
        logging.info("Loading openvino embedding model")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'mean_pooling': False,'normalize_embeddings': True, "batch_size": 4}
        ov = OpenVINOBgeEmbeddings( #using this method has the right value for query_instructions for bge models 
        model_name_or_path=model_name_or_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        show_progress=True,
        )
        return ov
    except Exception as e:
        raise ValueError("Error loading openvino embedding model{}".format(e))

def load_reranker_model() -> OpenVINOReranker:
    """
    Returns: 
        OpenVINOReranker: the loaded model.
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


def create_vectordb(docs, spliter_name, chunk_size, chunk_overlap, embedding_model, vector_store_name = vectorstore_path )-> str:
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
    #start_time = time.time()    
    #db = FAISS.from_documents(texts, load_hf_embedding_model())
    start_time = time.time()    
    db = FAISS.from_documents(texts, load_ov_embedding_model(embedding_model))
    print("--- %s seconds ---" % (time.time() - start_time))
    db.save_local(vector_store_name)

    return "Vectorstore created at {}".format(vector_store_name)

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
    prompt = PromptTemplate(input_variables=["QWEN_DEFAULT_RAG_PROMPT", "context", "question"], template=qwen_rag_prompt_template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return  create_retrieval_chain(retriever, combine_docs_chain)

#add the update retriever method


#print(create_vectordb([pdf_path], "RecursiveCharacter", 1024, 20))

#fatto prove con bge-m3 con 1000 di chunk size e 50 di chunk overlap e abbasando il score_threshold a 0.2 perchè con 0.5 come prima non trovava nulla
# ora prova ad rimettere a 400 il chunk size lasciando lo stesso score_trthershold e vedere se anche il retrive con bge-m3 si veloizza perchè per ora è troppo lento 
#  troppo lento il vector db con bge-m3 con 400 di chunk size e 50 di chunk overlap e score_threshold a 0.2 anche perchè è multi-lingua ma non ci interessa
