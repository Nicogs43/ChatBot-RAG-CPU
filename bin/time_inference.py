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

time_init_dict = {}
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
    
    start_time_initialization = time.time()
    # Load model-specific pipeline
    ov_slm = initialize_openvino_pipeline(ov_config, model_id=model_details["model_path"])
    
    # Set pipeline configurations
    pipeline_kwargs = dict(
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=0.7 > 0.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        tokenizer=ov_slm.pipeline.tokenizer,
        skip_special_tokens=True,
    )
    ov_slm.pipeline_kwargs = pipeline_kwargs
    
    # Create the RAG chain with model-specific configurations
    rag_chain = create_rag_chain(
        vector_index=vectorstore,
        slm=ov_slm,
        vector_search_top_k=5,
        vector_rerank_top_n=2,
        reranker=reranker,
        search_method="similarity_score_threshold",
        score_threshold=0.6,
        prompt_template=model_details["prompt_template"],
        default_rag_prompt=model_details["default_rag_prompt"],
    )

    end_time_initialization = time.time()
    time_init_dict[model_name] = end_time_initialization - start_time_initialization
    logging.info(f"Initialization time for model {model_name}: {time_init_dict[model_name]}")
    
    # Collect answers for this model
    time_inference_list = []
    try:
        for question in data['Question']:
            start_time_inference = time.time()
            output = rag_chain.invoke(input={"input": question})
            end_time_inference = time.time()
            time_inference_list.append(end_time_inference - start_time_inference)
            time.sleep(1)  # Pause to prevent overloading
        request_cancel(ov_slm=ov_slm)
    except KeyboardInterrupt as e:
        logging.info(e)
    finally:
        del ov_slm
        logging.info(f"Session ended for model {model_name}.")

    time_inference_list = pd.DataFrame(time_inference_list, columns=['inference_time'])
    time_inference_list.to_csv(f"../data/inference_time/{model_name}_time_inference_new.csv", index=False)
    print(f"Data saved successfully for model: {model_name}")
    

save_time_init = pd.DataFrame.from_dict(time_init_dict, "index", columns=["initialization_time"])
save_time_init.to_csv("../data/inference_time/time_initialization.csv", index=False)
# Clean up resources after all models have been tested
del vectorstore
del reranker
logging.info("All models tested. Resources cleaned up.")
