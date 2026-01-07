"""
谓词conflict(𝐶) 的实现：判断两个星型模式或一个星型内部（对应的json）是否存在以下8种矛盾类型之一：

借助llm谓词区分以下八种矛盾类型：
Negation - Negating the original sentence
Numeric - Number mismatch or number out of scope
Content - Changing one or multiple attributes of an event or entity
Perspective / View / Opinion - Inconsistency in one’s attitude/perspective/opinion
Emotion / Mood / Feeling - Inconsistency in one’s attitude/emotion/mood
Relation - Description of two mutually exclusive relations between entities
Factual - Need external world knowledge to confirm the contradiction
Causal - The effect does not match the cause

旧版
useless
"""
import json
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# 定义系统提示词，包含矛盾类型说明和示例
system_prompt = """
You are an expert in detecting contradiction types between two sentences.

Given two sentences, identify if they are contradictory and if so, classify the contradiction into one of the following 8 types:

Negation - Negating the original sentence
Numeric - Number mismatch or number out of scope
Content - Changing one or multiple attributes of an event or entity
Perspective / View / Opinion - Inconsistency in one’s attitude/perspective/opinion
Emotion / Mood / Feeling - Inconsistency in one’s attitude/emotion/mood
Relation - Description of two mutually exclusive relations between entities
Factual - Need external world knowledge to confirm the contradiction
Causal - The effect does not match the cause

If the sentences are NOT contradictory, output: No Contradiction

Output ONLY the contradiction type: one of ["Negation", "Numeric", "Content", "Perspective / View / Opinion", "Emotion / Mood / Feeling", "Relation", "Factual", "Causal"] or "No Contradiction".

No explanation or additional text.
"""


def classify_contradiction_type(pair, llm):
    prompt = f"""
Sentence 1: "{pair['sentence_1']}"
Sentence 2: "{pair['sentence_2']}"

What type of contradiction do these sentences represent?
"""
    response = llm.generate(
        user_message=prompt,
        system_message=system_prompt
    )
    return response.strip()

def detect_contradiction(pair, llm):
    prompt = f"""
Sentence 1: "{pair['sentence_1']}"
Sentence 2: "{pair['sentence_2']}"

Do these sentences contradict each other? 
Answer only "Yes" or "No".
"""
    system_prompt = (
        "You are a language reasoning assistant. "
        "Only answer 'Yes' if the sentences are contradictory, otherwise 'No'."
    )
    response = llm.generate(
        user_message=prompt,
        system_message=system_prompt
    )
    return response.strip().lower() == 'yes'

def main():
    pairs = [
        {"sentence_1": "Zully donated her kidney.", "sentence_2": "Zully never donated her kidney."},
        {"sentence_1": "All the donors are between 20 to 45 years old.", "sentence_2": "Lisa, who donates her kidney, she is 70 years old."},
        {"sentence_1": "Zully Broussard donated her kidney to a stranger.", "sentence_2": "Zully Broussard donated her kidney to her close friend."},
        {"sentence_1": "The doctor spoke highly of the project and called it “a breakthrough”.", "sentence_2": "The doctor disliked the project, saying it had no impact at all."},
        {"sentence_1": "The rescue team searched for the boy worriedly.", "sentence_2": "The rescue team searched for the boy happily."},
        {"sentence_1": "Jane and Tom are a married couple.", "sentence_2": "Jane is Tom’s sister."},
        {"sentence_1": "The road T51 was located in New York.", "sentence_2": "The road T51 was located in California."},
        {"sentence_1": "I slam the door.", "sentence_2": "After I do that, the door opens."}
    ]

    for pair in pairs:
        if detect_contradiction(pair, llm):
            contradiction_type = classify_contradiction_type(pair, llm)
        else:
            contradiction_type = "No Contradiction"
        print(f"Sentences:\n  1: {pair['sentence_1']}\n  2: {pair['sentence_2']}\nContradiction type: {contradiction_type}\n")


if __name__ == "__main__":
    main()

