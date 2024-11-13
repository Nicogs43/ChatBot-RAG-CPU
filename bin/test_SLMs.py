from config import phi_rag_prompt_template, ov_config, vectorstore_path, model_dict
import pandas as pd 
from vectordb import load_reranker_model, load_ov_embedding_model, create_rag_chain
from chatbot import request_cancel, initialize_openvino_pipeline
from config import vectorstore_path, ov_config, phi_rag_prompt_template, gemma_rag_prompt_template, qwen_rag_prompt_template
from langchain_community.vectorstores import FAISS
import warnings
import logging
import time

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)

# Load the question dataset
try:
    data = pd.read_csv('../data/questions_ground_truths_25.csv')
except FileNotFoundError:
    raise ValueError("Error loading data")

# Load the vector store
try:
    vectorstore = FAISS.load_local(vectorstore_path, embeddings=load_ov_embedding_model(), allow_dangerous_deserialization=True)
except Exception as e:
    raise ValueError("Error loading vectorstore{}".format(e))

logging.info("Loading reranker")
reranker = load_reranker_model()


# Iterate over each model in model_dict
for model_name, model_details in model_dict.items():
    logging.info(f"Testing model: {model_name}")
    
    # Load model-specific pipeline
    ov_llm = initialize_openvino_pipeline(ov_config, model_id=model_details["model_path"])
    
    # Set pipeline configurations
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
    )
    ov_llm.pipeline_kwargs = pipeline_kwargs
    
    # Create the RAG chain with model-specific configurations
    rag_chain = create_rag_chain(
        db=vectorstore,
        llm=ov_llm,
        vector_search_top_k=5,
        vector_rerank_top_n=2,
        reranker=reranker,
        search_method="similarity_score_threshold",
        score_threshold=0.6,
        prompt_template=model_details["prompt_template"],
        default_rag_prompt=model_details["default_rag_prompt"],
    )
    
    # Collect answers for this model
    answers = []
    try:
        for question in data['Question']:
            output = rag_chain.invoke(input={"input": question})
            answers.append(output['answer'])
            time.sleep(1)  # Pause to prevent overloading
        request_cancel(ov_llm=ov_llm)
    except KeyboardInterrupt as e:
        logging.info(e)
    finally:
        del ov_llm
        logging.info(f"Session ended for model {model_name}.")

    # Save results for this model
    data_with_answer = {
        "question": data['Question'],
        "answer": answers,
        "ground_truths": data['ground_truths']
    }
    save_data = pd.DataFrame.from_dict(data_with_answer)
    save_path = f"../data/{model_name}_data_with_answer.csv"
    save_data.to_csv(save_path, index=False)
    print(f"Data saved successfully for model {model_name} at {save_path}")

# Clean up resources after all models have been tested
del vectorstore
del reranker
logging.info("All models tested. Resources cleaned up.")
