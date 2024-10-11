import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

#
# Load the data
data = pd.read_json('../data/data_with_answer_and_contexts_cleaned.json')
# Create a dataset from the data
dataset = Dataset.from_pandas(data)
print(dataset)
print(dataset.features)


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
