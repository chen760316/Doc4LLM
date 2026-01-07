"""
逻辑contra(s,s')的实现
简单判断两个句子之间出否存在矛盾
可作为T(s, s′,\tau)的前置判断，若纯在矛盾，则继续判断矛盾的具体类型，否则无需判断矛盾类型（减少判断逻辑）
"""

import json
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
from itertools import combinations
import random

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

def contra(s1: str, s2: str, llm) -> bool:
    """
    Returns True if s1 and s2 contradict each other (semantically), using LLM.
    """
    system_prompt = (
        "You are a natural language inference engine. "
        "Determine whether the following two sentences contradict each other. "
        "Respond with only one word: 'Yes' if they contradict, or 'No' if they do not."
    )

    user_message = f"Sentence 1: {s1}\nSentence 2: {s2}\nDo they contradict?"

    try:
        response = llm.generate(
            user_message=user_message,
            system_message=system_prompt
        )
        answer = response.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"⚠️ Error in contra() check:\n{e}")
        return False  # fallback

def read_sentences_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        # 读取所有非空行，去除两端空白符
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

def build_pairs(sentences, max_pairs=None):
    all_pairs = list(combinations(sentences, 2))
    if max_pairs is None or max_pairs >= len(all_pairs):
        return all_pairs
    else:
        # 随机抽取 max_pairs 对，保证无重复
        return random.sample(all_pairs, max_pairs)


# pairs_to_check = [
#     (
#         "John Smith, a 45-year-old American software engineer, is married and resides in San Francisco.",
#         "John Smith is a 30-year-old Canadian graphic designer living in New York."
#     ),
#     (
#         "TechNova Inc., founded in 2012, operates in the AI research industry.",
#         "TechNova Inc. is a nonprofit healthcare organization founded in 2012."
#     ),
#     (
#         "Lily is a 12-year-old.",
#         "Lily is a 12-year-old girl who loves mathematics."
#     )
# ]

# 假设txt路径
filepath = "refined_node_descriptions.txt"

# 读句子
sentences = read_sentences_from_file(filepath)
print(f"读取到 {len(sentences)} 条句子。")

# 生成pairs
pairs_to_check = build_pairs(sentences, 10)
print(f"构造了 {len(pairs_to_check)} 对句子进行矛盾检测。")

for s1, s2 in pairs_to_check:
    result = contra(s1, s2, llm)
    status = "❌ CONTRADICTION" if result else "✅ NO CONTRADICTION"
    print(f"{status}\n→ S1: {s1}\n→ S2: {s2}\n")