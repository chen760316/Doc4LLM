"""
实现谓词𝑆(e𝑥, 𝑠)，将关系（边）转化为llm生成的自然语言
𝑠是一个字符串变量，用于文档中的子字符串
(a)𝑆(e𝑥, 𝑠)将子字符串𝑠与实体关系e𝑥绑定，这里的s由大模型生成
生成关系（边）的原始描述initial_relation_descriptions.txt和润色后的描述refined_relation_descriptions.txt
useful
"""

from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re

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


def refine_with_llm(sentences, llm, batch_size=1):
    """
    使用 LLM 对句子进行润色
    """
    refined = []
    system_prompt = (
        "You are a professional writer and editor. Your task is to rewrite the following sentence "
        "so that it sounds natural, fluent, and contextually appropriate in English. Ensure that:\n"
        "- All factual information is preserved accurately.\n"
        "- The sentence flows smoothly and reads like it was written by a native English speaker.\n"
        "- Relationship details (e.g. time, place, method) are clearly and concisely expressed.\n"
        "- Avoid redundancy or awkward phrasing.\n"
        "Return only the revised sentence without any explanations or extra commentary."
    )

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        for sentence in batch:
            try:
                response = llm.generate(
                    user_message=sentence,
                    system_message=system_prompt
                )
                refined.append(response.strip())
            except Exception as e:
                print(f"⚠️ Error processing sentence: {sentence}\n{e}")
                refined.append(sentence)  # fallback

    return refined

def build_relation_sentences(data):
    """
    将 relationship 字段和 edge_properties 拼成一整句话。
    """
    sentences = []
    for triple in data:
        base = triple["relationship"].strip()
        props = triple.get("edge_properties", {})
        if props:
            attrs = ", ".join(f"{k}: {v}" for k, v in props.items())
            sentence = f"{base}. (Attributes: {attrs})"
        else:
            sentence = base
        sentences.append(sentence)
    return sentences



# === 主程序 ===
if __name__ == "__main__":
    input_path = "../output_knowledge_graph.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 构造关系句子（关系 + 属性）
    relation_sentences = build_relation_sentences(data)

    # 保存原始
    with open("initial_relation_descriptions.txt", "w", encoding="utf-8") as f:
        for line in relation_sentences:
            f.write(line + "\n")

    # 2. 用 LLM 润色
    print("💬 Refining relationship descriptions...\n")
    refined = refine_with_llm(relation_sentences, llm)

    for line in refined:
        print("•", line)

    with open("refined_relation_descriptions.txt", "w", encoding="utf-8") as f:
        for line in refined:
            f.write(line + "\n")


