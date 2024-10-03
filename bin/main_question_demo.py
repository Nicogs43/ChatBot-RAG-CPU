from vectordb import load_reranker_model, load_ov_embedding_model, create_rag_chain
from chatbot import request_cancel, initialize_openvino_pipeline
from config import vectorstore_path, ov_config, questions
from langchain_community.vectorstores import FAISS
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
    try:
        # Loop through each question and get the response
        for query in questions:
            print(f"Question: {query}")
            start = time.time()
            output = rag_chain.invoke(input={"input": query})
            print("Time taken: ", time.time() - start)
            print("Answer:", output['answer'])
            print("-" * 100)  # Separator between each question/answer

    except KeyboardInterrupt:
        print("Session ended.")
    finally:
        del vectorstore
        del reranker
        request_cancel(ov_llm=ov_llm)
        print("Resources released.")


if __name__ == "__main__":
    main()
