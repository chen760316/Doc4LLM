"""
谓词M(𝑥. ¯𝐴,𝑦. ¯𝐵) | M(𝑒𝑥. ¯𝐴,𝑒𝑦. ¯𝐵)的实现
基于NLI的模型（判断语义冲突或蕴含）
借助大模型判断冲突性/矛盾性，映射为 [0,1] 评分
"""

from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

def compute_conflict_score_llm(premise, hypothesis):
    system_message = "You are a natural language inference assistant."
    user_message = f"""
Determine whether the following two sentences contradict each other.

Please return a single decimal number between 0 and 1 representing the probability of contradiction,
where 0 means no contradiction and 1 means strong contradiction.

Premise: "{premise}"
Hypothesis: "{hypothesis}"

Only return the decimal number.
"""

    response = llm.generate(system_message, user_message)

    # Assuming the response is a dict with the generated text in something like response["choices"][0]["message"]["content"]
    # Adjust this according to your actual return format.
    content = response.strip()  # if response is string

    try:
        score = float(content)
        score = max(0.0, min(1.0, score))
    except Exception as e:
        print("Failed to parse score, returning 0.0 by default.", e)
        score = 0.0

    return score


def classify_conflict(score, threshold_high=0.7, threshold_low=0.3):
    if score > threshold_high:
        return "❌ Strong contradiction"
    elif score < threshold_low:
        return "✅ No contradiction / Entailment"
    else:
        return "⚠️ Neutral / Possible contradiction"

# Test sentence pairs
test_cases = [
    ("The man is eating.", "The man is not eating."),
    ("The cat is sleeping on the couch.", "There is a cat on the sofa."),
    ("Tom was born in 1990", "Tom died in 2020"),
    ("Tom was born in 1990", "Tom died in 1980"),
]

for premise, hypothesis in test_cases:
    score = compute_conflict_score_llm(premise, hypothesis)
    label = classify_conflict(score)
    print(f"Premise: \"{premise}\"\nHypothesis: \"{hypothesis}\"\nConflict score: {score:.4f} → {label}\n")
