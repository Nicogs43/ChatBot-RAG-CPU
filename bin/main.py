from vectordb import load_hf_embedding_model, load_reranker_model
from chatbot import bot , request_cancel,initialize_openvino_pipeline
from config import vectorstore_path, ov_config
from langchain_community.vectorstores import FAISS
#import gradio as gr
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
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=load_hf_embedding_model(), allow_dangerous_deserialization=True)
    except Exception as e:
        raise ValueError("Error loading vectorstore{}".format(e))
    
    print("Loading reranker")
    reranker = load_reranker_model()

    print("Loading openvino pipeline")
    ov_llm = initialize_openvino_pipeline(ov_config)
    try:
        #take the input from the user inside a loop to keep the chatbot running
        while True:
            query = input("Insert here your question (type exit to quit): ")
            if query == "exit":
                if output:
                    request_cancel(ov_llm=ov_llm)
                break
            start = time.time()
            output = bot(
                vectorstore=vectorstore,
                query=query,
                ov_llm=ov_llm,
                vector_search_top_k=5,
                vector_rerank_top_n=2,
                reranker=reranker,
                search_method="similarity_score_threshold",
                score_threshold=0.5,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                hide_full_prompt=True,
            )
            print("Time taken: ", time.time() - start)
            print(output['answer'])
            request_cancel(ov_llm=ov_llm)

    except KeyboardInterrupt:
        print("Session ended.")
    finally:
        del vectorstore
        del reranker
        del ov_llm
        print("Resources released.")



"""
def chatbot_interface(query, history):
    if query.lower() == "exit":
        request_cancel()
        return history + [[query, "Session ended."]]

    # Generate bot response
    output = bot(
        vectorstore=vectorstore,
        query=query,
        vector_search_top_k=3,
        vector_rerank_top_n=2,
        run_rerank= True,
        search_method="similarity_score_threshold",
        score_threshold=0.2,
        temperature=0.5,
        top_p=0.9,
        top_k=50,
        repetition_penalty=0.9,
        hide_full_prompt=True,
    )
    
    # Append the user query and bot response to the history
    history = history + [[query, output]]
    
    # Optionally, call request_cancel if needed
    request_cancel()
    
    return history

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot App")
    chatbot = gr.Chatbot()
    state = gr.State([])
    txt = gr.Textbox(show_label=False, placeholder="Type your message here...")

    def user_interaction(user_input, history):
        history = history or []
        new_history = chatbot_interface(user_input, history)
        return "", new_history

    txt.submit(user_interaction, [txt, state], [txt, chatbot])
"""
if __name__ == "__main__":
    main()
    #demo.launch()