import pandas as pd
import logging
from datasets import Dataset
import json
from config import model_dict


def proc_datacreated_by_qwen():
    filtered_dict = {k: v for k, v in model_dict.items() if k.startswith('q')}
    for model_name, model_details in model_dict.items():
        df = pd.read_csv(f"../data/{model_name}_data_with_answer.csv")
        # Initialize lists to store the cleaned answers and contexts
        cleaned_answers = []
        contexts = []

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            answer_text = row['answer']
            
            # Split the answer text based on the markers <|im_start|> and <|im_end|>
            if "<|im_start|>user" in answer_text and "<|im_start|>assistant<|im_end|>" in answer_text:
                # Extract context and the cleaned answer
                context_part = answer_text.split("Context:")[1].split("<|im_end|>")[0].strip()
                answer_part = answer_text.split("<|im_start|>assistant<|im_end|>")[1].strip()
                
                # Append to the lists
                cleaned_answers.append(answer_part)
                contexts.append([context_part])
            else:
                # If markers are not found, append the original content
                cleaned_answers.append(answer_text)
                contexts.append([""])

        print(len(contexts))
        # Create a new dictionary with cleaned answers and contexts
        dictionary = {
            "question": df['question'].tolist(),
            "ground_truth": df['ground_truths'].tolist(),
            "answer": cleaned_answers,
            "contexts": contexts

        }
        try:
            with open(f'../data/{model_name}_with_answer_and_contexts_cleaned.json', 'w') as f:
                json.dump(dictionary, f)
        except Exception as e:
            logging.error("Error saving the data: {}".format(e))

    logging.info("Data cleaning completed successfully.")

