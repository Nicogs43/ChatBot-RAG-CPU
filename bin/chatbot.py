from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from vectordb import create_rag_chain
from transformers import TextIteratorStreamer
import logging

logging.basicConfig(level=logging.INFO)


#can be exploit using different configurations

def initialize_openvino_pipeline(ov_config, max_new_tokens=140):
    """
    Initialize the openvino pipeline
    Args:
        ov_config: openvino configuration
        max_new_tokens: maximum new tokens
    Returns: openvino pipeline
    """
    logging.info("Loading openvino pipeline")
    try:
        ov_llm = HuggingFacePipeline.from_model_id(
        model_id="../model/microsoft/Phi-3.5-mini-instruct/int4",
        task="text-generation",
        backend="openvino",
        model_kwargs={"device": "CPU", "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": max_new_tokens},

    )
    except Exception as e:
        raise ValueError("Error loading openvino pipeline{}".format(e))
    return ov_llm


def bot(
        vectorstore, ov_llm, query, vector_search_top_k, vector_rerank_top_n, search_method, score_threshold, temperature, top_p, top_k, repetition_penalty, hide_full_prompt, reranker=None):
    streamer = TextIteratorStreamer(
        ov_llm.pipeline.tokenizer,
        timeout=60.0,
        skip_prompt=hide_full_prompt,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    pipeline_kwargs = dict(
        max_new_tokens=1024,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    ov_llm.pipeline_kwargs = pipeline_kwargs

    rag_chain = create_rag_chain(
        vectorstore,
        ov_llm,
        vector_search_top_k,
        vector_rerank_top_n,
        reranker,
        search_method,
        score_threshold,
    )
    response = rag_chain.invoke(input={"input": query})
    return response
    

def request_cancel(ov_llm):
    ov_llm.pipeline.model.request.cancel()
    
