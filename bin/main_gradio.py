import argparse
import time

import gradio as gr

from chatbot import RagEngine, load_retrieval_resources
from config import default_model, model_dict

engine = None  # created in __main__, once the models are loaded


# Define the chatbot response function
def chatbot_response(user_input, history):
    start_time = time.time()
    answer = ""
    try:
        for piece in engine.stream(user_input):
            answer += piece
            yield answer
    except Exception as e:
        yield f"An error occurred: {e}"
    finally:
        print(f"Time taken: {time.time() - start_time} seconds")


with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=chatbot_response,
        chatbot=gr.Chatbot(placeholder="<strong> Ask a question about the maritime domain <strong>", height=600),
        title="MarineChat 🌬️⛵",
        examples=["What are the typical winds in Genoa?", "What is the VHF channel of the port of Imperia?", "Can you tell me the email address of the port of Ischia?", "What is the depth of the sea in the port of Alassio?"],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarineChat Gradio UI")
    parser.add_argument("--model", choices=list(model_dict), default=default_model, help="Model key from config.model_dict")
    args = parser.parse_args()

    # Initialize models and resources
    vectorstore, reranker = load_retrieval_resources()
    engine = RagEngine(args.model, vectorstore, reranker)

    demo.launch(inbrowser=True, theme=gr.themes.Soft())  # , share=True, auth=auth_check)
