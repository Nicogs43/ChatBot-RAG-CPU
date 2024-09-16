DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

rag_prompt_template = f"""<|system|> {DEFAULT_RAG_PROMPT }<|end|>""" + """
<|user|>
Question: {input} 
Context: {context} 
<|end|>
Answer: 
<|assistant|>"""

vectorstore_path = "../vectorstore"

ov_config = {
    "KV_CACHE_PRECISION": "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}

pdf_path = "../samples/Ports_chapters_10_to_20.pdf"