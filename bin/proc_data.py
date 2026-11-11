import json
import logging

import pandas as pd

from config import DATA_DIR, model_dict


def _split_legacy_answer(answer_text):
    """Thesis-era CSVs: the answer column holds the whole echoed Qwen prompt.
    Returns (clean answer, context) parsed out of the chat markers."""
    if "<|im_start|>user" in answer_text and "<|im_start|>assistant<|im_end|>" in answer_text:
        context_part = answer_text.split("Context:")[1].split("<|im_end|>")[0].strip()
        answer_part = answer_text.split("<|im_start|>assistant<|im_end|>")[1].strip()
        return answer_part, [context_part]
    # If markers are not found, keep the original content
    return answer_text, [""]


def proc_datacreated_by_qwen():
    for model_name in model_dict:
        df = pd.read_csv(DATA_DIR / f"{model_name}_data_with_answer.csv")

        if "contexts" in df.columns:
            # Current test_SLMs.py output: clean answers plus a JSON list of contexts
            cleaned_answers = df["answer"].tolist()
            contexts = [json.loads(c) for c in df["contexts"]]
        else:
            cleaned_answers, contexts = [], []
            for answer_text in df["answer"]:
                answer, context = _split_legacy_answer(answer_text)
                cleaned_answers.append(answer)
                contexts.append(context)

        print(len(contexts))
        # Create a new dictionary with cleaned answers and contexts
        dictionary = {
            "question": df["question"].tolist(),
            "ground_truth": df["ground_truths"].tolist(),
            "answer": cleaned_answers,
            "contexts": contexts,
        }
        try:
            with open(DATA_DIR / f"{model_name}_with_answer_and_contexts_cleaned.json", "w") as f:
                json.dump(dictionary, f)
        except Exception as e:
            logging.error("Error saving the data: {}".format(e))

    logging.info("Data cleaning completed successfully.")


if __name__ == "__main__":
    proc_datacreated_by_qwen()
