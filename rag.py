from dotenv import load_dotenv
import os
from fastapi import FastAPI
import dspy
load_dotenv()
from dspy.evaluate import SemanticF1
import ujson
api_key = os.getenv("OPENAI_API_KEY")
lm = dspy.LM('openai/gpt-4o-mini', api_key=api_key)
dspy.configure(lm=lm)
from pydantic import BaseModel
class Question(BaseModel):
    question: str

cot = dspy.ChainOfThought('question -> response')
with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]

data = [dspy.Example(**d).with_inputs('question') for d in data]

# Let's pick an `example` here from the data.
example = data[4]
print(example)

# Instantiate the metric.
metric = SemanticF1(decompositional=True)

# Produce a prediction from our `cot` module, using the `example` above as input.
pred = cot(**example.inputs())

# Compute the metric score for the prediction.
score = metric(example, pred)

print(f"Question: \t {example.question}\n")
print(f"Gold Response: \t {example.response}\n")
print(f"Predicted Response: \t {pred.response}\n")
print(f"Semantic F1 Score: {score:.2f}")
