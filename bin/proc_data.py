import pandas as pd
import logging
from datasets import Dataset
import json

df = pd.read_csv('../data/data_with_answer.csv')

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
#contexts = [contexts]
#print(contexts)
# Create a new dictionary with cleaned answers and contexts
dictionary = {
    "question": df['question'].tolist(),
    "answer": cleaned_answers,
    "contexts": contexts,
    "ground_truth": df['ground_truths'].tolist()

}

#df = pd.DataFrame.from_dict(dictionary)
#df.to_csv('../data/data_with_answer_and_contexts_cleaned.csv', index=False)
# Print the first answer and context to verify

try:
    with open('../data/data_with_answer_and_contexts_cleaned.json', 'w') as f:
        json.dump(dictionary, f)
except Exception as e:
    logging.error("Error saving the data: {}".format(e))


"""
dataset = Dataset.from_dict(dictionary)
print(dataset)
print(dataset.features)

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

result = evaluate(
    dataset=dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()

df = pd.concat(result, ignore_index=True)
df.to_csv('../data/evaluation_result.csv', index=False)

"""