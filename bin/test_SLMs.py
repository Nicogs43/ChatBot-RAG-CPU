import argparse
import gc
import json
import logging
import time
from pathlib import Path

import pandas as pd

from chatbot import RagEngine, load_retrieval_resources
from config import DATA_DIR, QUESTIONS_PATH, model_dict

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Answer the evaluation questions with each model in model_dict")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(model_dict),
        default=list(model_dict),
        help="Model keys to test (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Where to write <model>_data_with_answer.csv (default: data/, which holds the thesis results)",
    )
    args = parser.parse_args()

    # Load the question dataset
    try:
        data = pd.read_csv(QUESTIONS_PATH)
    except FileNotFoundError:
        raise ValueError(f"Question dataset not found: {QUESTIONS_PATH}") from None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    vectorstore, reranker = load_retrieval_resources()

    for model_name in args.models:
        logging.info(f"Testing model: {model_name}")
        engine = RagEngine(model_name, vectorstore, reranker)

        # Collect answers (and the retrieved contexts) for this model
        answers, contexts = [], []
        try:
            for question in data["Question"]:
                output = engine.invoke(question)
                answers.append(output["answer"])
                contexts.append(json.dumps([doc.page_content for doc in output["context"]]))
                time.sleep(1)  # Pause to prevent overloading
        except KeyboardInterrupt as e:
            logging.info(e)
        finally:
            engine = None
            gc.collect()
            logging.info(f"Session ended for model {model_name}.")

        if len(answers) < len(data):
            logging.info(f"Model {model_name} interrupted after {len(answers)} answers; nothing saved.")
            break

        # Save results for this model
        save_data = pd.DataFrame(
            {
                "question": data["Question"],
                "answer": answers,
                "ground_truths": data["ground_truths"],
                "contexts": contexts,  # JSON list of the retrieved passages
            }
        )
        save_path = args.output_dir / f"{model_name}_data_with_answer.csv"
        save_data.to_csv(save_path, index=False)
        print(f"Data saved successfully for model {model_name} at {save_path}")

    logging.info("All models tested. Resources cleaned up.")


if __name__ == "__main__":
    main()
