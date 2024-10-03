DEFAULT_RAG_PROMPT = """\
You are an assistant for question-answering tasks about maritime domain. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\
"""

QWEN_DEFAULT_RAG_PROMPT = """\
You are a knowledgeable assistant specializing in answering questions about ports, sailing, and maritime activities along the Italian peninsula. Use the most relevant pieces of the retrieved context below to support your answer. If you don't know the answer or the context is insufficient, state that you are unsure. Use a maximum of three sentences, and keep your answer concise, accurate, and where applicable, use maritime terminology."""

phi_rag_prompt_template = f"""<|system|> 
{DEFAULT_RAG_PROMPT }<|end|>""" + """
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
<|im_start|>assistant<|im_end|>
"""

vectorstore_path = "../vectorstore_ov_400_50"

ov_config = {
    "KV_CACHE_PRECISION": "u8",
    "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32",
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": "",
}

pdf_path = "../samples/Ports_chapters_10_to_20.pdf"

# List of predefined questions
questions = [
    "What are the typical winds in Genoa?",
    "What is the depth of the sea in the port of Alassio?",
    "What is the VHF channel of the port of Imperia?",
    "Where is situated the Golfo Marconi?",
    "Can you tell me the email address of the port of Ischia?",
    "Can you illustrate how one should enter the port of Genoa, what instructions should I follow and what should I watch out for?"
]