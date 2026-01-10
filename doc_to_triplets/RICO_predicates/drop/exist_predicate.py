"""
谓词exist(𝑥,𝑦)的实现，判断实体x和实体y之间是否存在连接边
想借助llm从原始文本中提取实体，随机组成实体对，然后判断实体对在output_knowledge_graph.json中是否存在连接关系
"""

import json
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import os
import random

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

text = """John Smith, a 45-year-old software engineer from San Francisco, works at TechNova Inc., a leading AI research company founded in 2012. He has been leading the Natural Language Processing (NLP) team there since 2018.

John's wife, Emily Clark, is a 42-year-old physician currently working at St. Mary's Hospital. They got married in 2010 and have two children: Lily, aged 12, and Max, aged 9.

In 2021, John gave a keynote speech at the International AI Conference held in Berlin, where he presented a paper on ethical machine learning. He also collaborates with Professor Alan Cheng from Stanford University on explainable AI research.

During the COVID-19 pandemic in 2020, Emily volunteered in the ICU department for over six months, earning a community award for outstanding medical service.

They currently live in a townhouse near Golden Gate Park and frequently donate to the San Francisco Education Fund, supporting STEM education for underprivileged students."""

# 读取 output_knowledge_graph.json 文件
with open('../../output_knowledge_graph.json', 'r', encoding='utf-8') as f:
    knowledge_graph = json.load(f)


def exist(x_name: str, y_name: str, graph: list) -> bool:
    """
    判断在知识图谱中实体 x_name 和 y_name 是否通过一条边连接
    """

    for triple in graph:
        node1 = triple["node_1"]
        node2 = triple["node_2"]

        # 简单匹配实体名即可（更复杂的匹配可以加 label 判断）
        name1 = node1.get("name", "")
        name2 = node2.get("name", "")

        if (name1 == x_name and name2 == y_name) or (name1 == y_name and name2 == x_name):
            return True  # 存在连接边

    return False  # 不存在连接边

def extract_named_entities(text, llm):
    system_prompt = (
        "You are an information extraction assistant. "
        "Your task is to extract all explicitly named entities mentioned in the text. "
        "Return the result in JSON format as a list of entities, each with a 'name' and a 'type'. "
        "The type must be one of: 'Person', 'Organization', 'Location', 'Event', or 'Other'."
    )

    user_prompt = f"""
Please extract named entities from the following text. Return JSON only.

Text:
\"\"\"
{text}
\"\"\"
"""

    response = llm.generate(
        user_message=user_prompt,
        system_message=system_prompt
    )

    # 🔧 去除Markdown代码块标记
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response.removeprefix("```json").strip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response.removesuffix("```").strip()

    try:
        entities = json.loads(cleaned_response)
        return entities
    except json.JSONDecodeError:
        print("⚠️ JSON 解析失败，原始响应如下：")
        print(response)
        return []


def main():
    print("*"*100)

    # 假设 llm 是你的 OpenAIClient/gpt-5.2 实例
    entities = extract_named_entities(text, llm)

    # 打印提取到的实体
    for ent in entities:
        print(f"📌 {ent['name']} ({ent['type']})")

    # 获取实体名称列表
    entity_names = [ent["name"] for ent in entities]

    # 随机抽取实体对
    random.seed(42)
    num_pairs = 5
    entity_pairs = random.sample([(a, b) for i, a in enumerate(entity_names) for b in entity_names[i + 1:]], num_pairs)

    # 检查连接
    print("\n🔗 检查实体连接：")
    for x, y in entity_pairs:
        result = exist(x, y, knowledge_graph)
        print(f"➡️  {x} ↔ {y} → {'✅ 存在连接' if result else '❌ 无连接'}")

if __name__ == "__main__":
    # 示例调用
    print(exist("John Smith", "Emily Clark", knowledge_graph))  # True（配偶关系）
    print(exist("John Smith", "TechNova Inc.", knowledge_graph))  # True（工作关系）
    print(exist("Emily Clark", "Lily", knowledge_graph))  # False（无直接边）
    main()
