import pandas as pd 
from vectordb import load_reranker_model, load_ov_embedding_model, create_rag_chain
from chatbot import request_cancel, initialize_openvino_pipeline
from config import vectorstore_path, ov_config, phi_rag_prompt_template, model_dict
from langchain_community.vectorstores import FAISS
import warnings

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)
import logging
import time
 
try:
    data = pd.read_csv('../data/questions_ground_truths.csv')
    print(data.head())
except FileNotFoundError:
    raise ValueError("Error loading data")

# Load the vector store
try:
    vectorstore = FAISS.load_local(vectorstore_path, embeddings=load_ov_embedding_model(), allow_dangerous_deserialization=True)
except Exception as e:
    raise ValueError("Error loading vectorstore{}".format(e))

try:
    logging.info("Loading reranker")
    reranker = load_reranker_model()
except Exception as e:
    logging.error(f"Error loading reranker model: {e}")
    raise ValueError("Error loading reranker model")

if not vectorstore or not reranker:
    raise ValueError("Error loading vectorstore or reranker model")


logging.info("Loading openvino pipeline")

ov_llm = initialize_openvino_pipeline(ov_config,model_id="../model/microsoft/Phi-3.5-mini-instruct/int4")
pipeline_kwargs = dict(
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=0.7 > 0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        tokenizer=ov_llm.pipeline.tokenizer,
        return_full_text=False,
        hide_full_prompt=True,
        skip_special_tokens=True,
        skip_prompt=True,
        #streamer=streamer,
    )
ov_llm.pipeline_kwargs = pipeline_kwargs
rag_chain = create_rag_chain(
        db=vectorstore,
        llm=ov_llm,
        vector_search_top_k=5,
        vector_rerank_top_n=2,
        reranker=reranker,
        search_method="similarity_score_threshold",
        score_threshold=0.6,
        prompt_template=phi_rag_prompt_template,
        default_rag_prompt="DEFAULT_RAG_PROMPT",
)


answers = []

try:
    for question in data['Question']:
        output = rag_chain.invoke(input = {"input": question})
        answers.append(output['answer'])
        time.sleep(1)
    request_cancel(ov_llm=ov_llm)
except KeyboardInterrupt as e:
    logging.info(e)
finally:
    del vectorstore
    del reranker
    del ov_llm
    logging.info("Session ended.")

data_with_answer= {
    "question": data['Question'],
    "answer": answers,
    "ground_truths": data['ground_truths']
}
save_data = pd.DataFrame.from_dict(data_with_answer)
save_data.to_csv('../data/data_with_answer.csv', index=False)
print("Data saved successfully")
