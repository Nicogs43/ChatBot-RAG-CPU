from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from vectordb import create_rag_chain
from transformers import TextIteratorStreamer
import logging
import warnings
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)

logging.basicConfig(level=logging.INFO)

def initialize_openvino_pipeline(ov_config, model_id = "../model/Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=140):
    """
    Initialize the openvino pipeline
    Args:
        ov_config: openvino configuration
        max_new_tokens: maximum new tokens
    Returns: openvino pipeline
    """
    logging.info("Loading openvino pipeline")
    try:
        ov_slm = HuggingFacePipeline.from_model_id(
        model_id= model_id,
        task="text-generation",
        backend="openvino",
        model_kwargs={"device": "CPU", "ov_config": ov_config},
        pipeline_kwargs={"max_new_tokens": max_new_tokens},
    )
        logging.info("Openvino model: {}".format(ov_slm.model_id))
    except Exception as e:
        raise ValueError("Error loading openvino pipeline{}".format(e))
    return ov_slm

def request_cancel(ov_slm):
    ov_slm.pipeline.model.request.cancel()
    