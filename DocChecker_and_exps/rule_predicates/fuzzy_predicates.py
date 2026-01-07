"""
谓词 contra(𝑠,𝑠′)的扩展实现形式，在冲突和不冲突之外扩展了pair对的关系，具体关系可分为等价，对立和不清楚

借助llm谓词区分模糊语句的具体语义（以下三种之一）
"- Equivalent: The sentences express the same core meaning, even if the wording or tone differs.\n"
"- Contradictory: The sentences express clearly opposing or logically incompatible meanings.\n"
"- Neither: The semantic relationship is unclear, partially overlapping, or unrelated.\n\n"
useful
"""
import json
from knowledge_graph_maker import OpenAIClient
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# 从文本文件读取句子对
ambiguous_pairs = []
with open("fuzzy_pairs.txt", "r", encoding="utf-8") as f:
    for line in f:
        s1, s2 = line.strip().split(" ||| ")
        ambiguous_pairs.append({"sentence_1": s1, "sentence_2": s2})

# LLM 判断语义关系
def classify_semantic_relation_with_reason(pair, llm):
    system_prompt = (
        "You are a language reasoning assistant. Your job is to evaluate whether two English sentences, "
        "which may contain vague, euphemistic, or ambiguous language, are semantically equivalent, contradictory, or neither.\n\n"

        "Please follow these definitions strictly:\n"
        "- Equivalent: The sentences express the same core meaning, even if the wording or tone differs.\n"
        "- Contradictory: The sentences express clearly opposing or logically incompatible meanings.\n"
        "- Neither: The semantic relationship is unclear, partially overlapping, or unrelated.\n\n"

        "Consider that natural language allows for variation in tone, intent, or clarity. "
        "You must judge based on the underlying **meaning**, not just surface words.\n\n"

        "After giving your judgment, provide a **brief explanation** of why you believe the two sentences are equivalent, contradictory, or neither.\n\n"

        "Here are some examples:\n"
        "Example 1:\n"
        "Sentence 1: \"He was let go from his position.\"\n"
        "Sentence 2: \"He resigned from the job.\"\n"
        "Label: Contradictory\n"
        "Reason: The first sentence implies that he was dismissed involuntarily, while the second suggests he left voluntarily.\n\n"

        "Example 2:\n"
        "Sentence 1: \"She is not unattractive.\"\n"
        "Sentence 2: \"She is beautiful.\"\n"
        "Label: Equivalent\n"
        "Reason: Both sentences communicate that she is attractive, though one uses a negation and the other a direct statement.\n\n"

        "Example 3:\n"
        "Sentence 1: \"He didn’t attend the meeting.\"\n"
        "Sentence 2: \"He skipped the meeting intentionally.\"\n"
        "Label: Neither\n"
        "Reason: The first sentence is neutral regarding intent, while the second implies intent, making the relationship unclear.\n\n"

        "Based on this, analyze the following sentence pair carefully and output **only one label** and a **brief explanation** of your reasoning:\n"
        "- Equivalent\n"
        "- Contradictory\n"
        "- Neither\n"
        "Explanation: <Brief explanation of the reasoning>"
    )

    user_prompt = f"""Sentence 1: "{pair['sentence_1']}"
Sentence 2: "{pair['sentence_2']}"
What is the semantic relationship? Please explain your reasoning."""

    response = llm.generate(
        user_message=user_prompt,
        system_message=system_prompt
    )
    return response.strip()


# 运行判断并打印结果
for pair in ambiguous_pairs:
    label = classify_semantic_relation_with_reason(pair, llm)
    print(f"Sentence 1: {pair['sentence_1']}")
    print(f"Sentence 2: {pair['sentence_2']}")
    print(f"LLM Classification: {label}\n")
