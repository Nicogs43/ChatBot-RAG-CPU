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

gemma_rag_prompt_template = f"""
{DEFAULT_RAG_PROMPT},""" + """<start_of_turn>user
{input}<end_of_turn><start_of_turn>context
{context}<end_of_turn>
<start_of_turn>model"""

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

model_dict = {
    "gemma": {
        "model_path": "../model/google/gemma-2-2b-it",
        "prompt_template": gemma_rag_prompt_template,
        "default_rag_prompt": DEFAULT_RAG_PROMPT
    },
    "phi_int4": {
        "model_path": "../model/microsoft/Phi-3.5-mini-instruct/int4",
        "prompt_template": phi_rag_prompt_template,
        "default_rag_prompt": DEFAULT_RAG_PROMPT
    },
    "phi_int8": {
        "model_path": "../model/microsoft/Phi-3.5-mini-instruct/int8",
        "prompt_template": phi_rag_prompt_template,
        "default_rag_prompt": DEFAULT_RAG_PROMPT
    },
    "qwen_0.5b": {
        "model_path": "../model/Qwen/Qwen2.5-0.5B-Instruct",
        "prompt_template": qwen_rag_prompt_template,
        "default_rag_prompt": QWEN_DEFAULT_RAG_PROMPT
    },
    "qwen_1.5b": {
        "model_path": "../model/Qwen/Qwen2.5-1.5B-Instruct",
        "prompt_template": qwen_rag_prompt_template,
        "default_rag_prompt": QWEN_DEFAULT_RAG_PROMPT
    },
    "qwen_3b": {
        "model_path": "../model/Qwen/Qwen2.5-3B-Instruct/int8",
        "prompt_template": qwen_rag_prompt_template,
        "default_rag_prompt": QWEN_DEFAULT_RAG_PROMPT
    }
}
