"""
实现谓词𝑆(𝑥, 𝑠)
𝑠是一个字符串变量，用于文档中的子字符串
(a)𝑆(𝑥, 𝑠)将子字符串𝑠与实体𝑥绑定，这里的𝑠由大模型生成
生成节点的原始描述initial_node_descriptions.txt和润色后的描述refined_node_descriptions.txt
useful
"""

import json
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

def generate_general_descriptions(json_data):
    from collections import OrderedDict

    nodes = OrderedDict()  # 保持顺序 + 去重

    for entry in json_data:
        for key in ["node_1", "node_2"]:
            node = entry[key]
            name = node.get("name", node.get("label", "Unknown"))

            # 合并属性（后者覆盖前者）
            if name not in nodes:
                nodes[name] = node.copy()
            else:
                nodes[name].update(node)

    descriptions = []

    for name, attrs in nodes.items():
        label = attrs.get("label", "Entity")
        rest_attrs = {k: v for k, v in attrs.items() if k not in {"name", "label"}}

        if rest_attrs:
            props_str = ", ".join(f"{k} is {v}" for k, v in rest_attrs.items())
            sentence = f"{name} is a {label} with the following attributes: {props_str}."
        else:
            sentence = f"{name} is a {label}."
        descriptions.append(sentence)

    return descriptions

def refine_with_llm(sentences, llm, batch_size=1):
    refined = []
    system_prompt = (
        "You are a professional writer and editor. Your task is to rewrite the following sentence "
        "so that it sounds natural, fluent, and contextually appropriate in English. Ensure that:\n"
        "- All factual information is preserved accurately.\n"
        "- The sentence flows smoothly and reads like it was written by a native English speaker.\n"
        "- Attributes such as age, occupation, location, affiliation, and status are clearly and concisely expressed.\n"
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
                refined.append(sentence)  # fallback to original

    return refined

# 示例调用
if __name__ == "__main__":

    input_path = "../output_knowledge_graph.json"  # ← 你可以根据实际情况修改文件路径

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    basic_sentences = generate_general_descriptions(data)

    for line in basic_sentences:
        print("•", line)

    with open("initial_node_descriptions.txt", "w", encoding="utf-8") as f:
        for sent in basic_sentences:
            f.write(sent + "\n")

    print("💬 Refining with LLM...\n")
    refined_sentences = refine_with_llm(basic_sentences, llm)

    for sent in refined_sentences:
        print("•", sent)

    # 保存结果
    with open("refined_node_descriptions.txt", "w", encoding="utf-8") as f:
        for sent in refined_sentences:
            f.write(sent + "\n")