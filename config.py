PHI_DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks about maritime domain. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

QWEN_DEFAULT_RAG_PROMPT = """\
You are a useful assistant for answering questions about ports and the sea in general along the Italian peninsula. Use the items shown in context below to answer the question shown also below. Use a maximum of three sentences and keep your answer concise."""

phi_rag_prompt_template = f"""<|system|> 
{PHI_DEFAULT_RAG_PROMPT }<|end|>""" + """
<|user|>
Question: {input} 
Context: {context} <|end|>
<|assistant|>"""

qwen_rag_prompt_template = f"""<|im_start|>system
{QWEN_DEFAULT_RAG_PROMPT }<|im_end|>""" + """
<|im_start|>user
Question: {input} 
Context: {context} 
<|im_end|>
<|im_start|>assistant<|endoftext|>
"""

vectorstore_path = "../vectorstore"

ov_config = {
    "KV_CACHE_PRECISION": "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}

pdf_path = "../samples/Ports_chapters_10_to_20.pdf"