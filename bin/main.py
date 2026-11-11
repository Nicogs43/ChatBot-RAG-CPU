import argparse
import time

from chatbot import RagEngine, load_retrieval_resources
from config import model_dict


def main():
    parser = argparse.ArgumentParser(description="MarineChat command-line chat")
    parser.add_argument(
        "--model", choices=list(model_dict), default="phi_int4", help="Model key from config.model_dict"
    )
    args = parser.parse_args()

    vectorstore, reranker = load_retrieval_resources()
    engine = RagEngine(args.model, vectorstore, reranker)

    try:
        # take the input from the user inside a loop to keep the chatbot running
        while True:
            query = input("Insert here your question (type exit to quit): ")
            if query == "exit":
                break
            start = time.time()
            for piece in engine.stream(query):
                print(piece, end="", flush=True)
            print("\nTime taken: ", time.time() - start)
            print("-" * 100)  # Separator between each question/answer
    except KeyboardInterrupt:
        engine.cancel()
        print("\nSession ended.")
    finally:
        del engine, vectorstore, reranker
        print("Resources released.")


if __name__ == "__main__":
    main()
