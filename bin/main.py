from vectordb import  load_reranker_model, load_ov_embedding_model, load_hf_embedding_model
from chatbot import  request_cancel,initialize_openvino_pipeline
from config import vectorstore_path, ov_config , phi_rag_prompt_template, qwen_rag_prompt_template
from langchain_community.vectorstores import FAISS
from vectordb import create_rag_chain
import warnings

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)
import time


def main():
    print("Loading vectorstore")
    print(vectorstore_path)
    
    output = None
    try:
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=load_ov_embedding_model(), allow_dangerous_deserialization=True)
    except Exception as e:
        raise ValueError("Error loading vectorstore{}".format(e))
    
    print("Loading reranker")
    reranker = load_reranker_model()

    print("Loading openvino pipeline")
    ov_slm = initialize_openvino_pipeline(ov_config, model_id = "../model/microsoft/Phi-3.5-mini-instruct/int4")
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
    rag_chain = create_rag_chain(
        vector_index=vectorstore,
        slm=ov_slm,
        vector_search_top_k=5,
        vector_rerank_top_n=2,
        reranker=reranker,
        search_method="similarity_score_threshold",
        score_threshold=0.6,
        prompt_template=phi_rag_prompt_template,
        default_rag_prompt="DEFAULT_RAG_PROMPT",
    )
    try:
        #take the input from the user inside a loop to keep the chatbot running
        while True:
            query = input("Insert here your question (type exit to quit): ")
            if query == "exit":
                if output:
                    request_cancel(ov_slm=ov_slm)
                break
            start = time.time()
            output = rag_chain.invoke(input={"input": query})
            print("Time taken: ", time.time() - start)
            print(output['answer'])
            print("-" * 100)  # Separator between each question/answer
            request_cancel(ov_slm=ov_slm)

    except KeyboardInterrupt:
        print("Session ended.")
    finally:
        del vectorstore
        del reranker
        del ov_slm
        print("Resources released.")


if __name__ == "__main__":
    main()