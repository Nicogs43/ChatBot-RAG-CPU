import gradio as gr
from vectordb import load_hf_embedding_model, load_reranker_model
from chatbot import bot, request_cancel, initialize_openvino_pipeline
from config import vectorstore_path, ov_config
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
        request_cancel(ov_llm=ov_llm)
        print(f"Time taken: {time.time() - start_time} seconds")
    return answer

def auth_check(username, password):
    #check if the user is authorized
    if username == 'admin' and password == 'tesi2024':
        return True
    else:
        return False



# Create a Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.ChatInterface(
        fn=chatbot_response,
        chatbot= gr.Chatbot(placeholder="<strong> Ask a question about the maritime domain <strong>", height=600),
        title="Maritime R.A.G. Chatbot 🌬️⛵",
        show_progress='full',
    )

# Run the Gradio interface
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

    rag_chain = bot(
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
    )
    demo.launch(inbrowser=True, share=True, auth=auth_check)

    del vectorstore
    del reranker
    #cleanup()

    #auto_reload=True, provare anche quwesto e il share=True