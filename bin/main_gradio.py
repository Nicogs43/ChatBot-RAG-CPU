import gradio as gr
from vectordb import load_hf_embedding_model, load_reranker_model, create_rag_chain
from chatbot import request_cancel, initialize_openvino_pipeline
from config import vectorstore_path, ov_config, qwen_rag_prompt_template
from langchain_community.vectorstores import FAISS
import warnings
import time

warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set",
    category=FutureWarning,
)

# Define the chatbot response function
def chatbot_response(user_input, history=""):
    start_time = time.time()
    try:
        output = rag_chain.invoke(input={"input": user_input})
        answer = output["answer"]
    except Exception as e:
        answer = f"An error occurred: {e}"
    finally:
        request_cancel(ov_slm=ov_llm)
        print(f"Time taken: {time.time() - start_time} seconds")
    return answer

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.ChatInterface(
        fn=chatbot_response,
        chatbot= gr.Chatbot(placeholder="<strong> Ask a question about the maritime domain <strong>", height=600),
        title="MarineChat 🌬️⛵",
        show_progress='full',
        examples=["What are the typical winds in Genoa?", "What is the VHF channel of the port of Imperia?", "Can you tell me the email address of the port of Ischia?","What is the depth of the sea in the port of Alassio?"],
    )
if __name__ == "__main__":
# Initialize models and resources
    print("Loading vectorstore")
    print(vectorstore_path)
    try:
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=load_hf_embedding_model(),
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        raise ValueError(f"Error loading vectorstore: {e}")

    print("Loading reranker")
    reranker = load_reranker_model()

    print("Loading OpenVINO pipeline")
    ov_llm = initialize_openvino_pipeline(ov_config)

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
    rag_chain = create_rag_chain(
        vector_index=vectorstore,
        slm=ov_llm,
        vector_search_top_k=5,
        vector_rerank_top_n=2,
        reranker=reranker,
        search_method="similarity_score_threshold",
        score_threshold=0.6,
        prompt_template=qwen_rag_prompt_template,
        default_rag_prompt="DEFAULT_RAG_PROMPT",

    )
    demo.launch(inbrowser=True)#, share=True, auth=auth_check)

    del vectorstore
    del reranker