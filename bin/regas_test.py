import pandas as pd 
from vectordb import load_reranker_model, load_ov_embedding_model, create_rag_chain
from chatbot import request_cancel, initialize_openvino_pipeline
from config import vectorstore_path, ov_config
from langchain_community.vectorstores import FAISS
from datasets import Dataset
import warnings
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)
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

print("Loading reranker")
reranker = load_reranker_model()

print("Loading openvino pipeline")
ov_llm = initialize_openvino_pipeline(ov_config)
"""rag_chain = bot(
                vectorstore=vectorstore,
                ov_llm=ov_llm,
                vector_search_top_k=5,
                vector_rerank_top_n=2,
                reranker=reranker,
                search_method="similarity_score_threshold",
                score_threshold=0.6,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                hide_full_prompt=True,
            )"""
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
)

answers = []
contexts = []

# traversing each question and passing into the chain to get answer from the system
retriever_test= vectorstore.as_retriever(search_kwargs={"top_k": 5, "score_threshold": 0.6}, search_method="similarity_score_threshold")
for question in data['Question']:
    answers.append(rag_chain.invoke(input = {"input": question}))
    contexts.append([docs.page_content for docs in retriever_test.get_relevant_documents(question)])

# Preparing the dataset
data_with_answer_and_text = {
    "question": data['Question'],
    "answer": answers,
    "contexts": contexts,
    "ground_truths": data['ground_truths']
}

dataset = Dataset.from_dict(data_with_answer_and_text)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset=dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()


print(df)