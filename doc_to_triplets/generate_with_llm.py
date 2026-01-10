"""
返回固定模式
暂时useless
"""
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

system_prompt = (
    "Extract entities and their relationships from the following text. "
    "For each entity, include attributes like age, occupation, location if available. "
    "For each relationship, include attributes like time, place, manner. "
    "Return a JSON array of objects with this schema:\n"
    "[\n"
    "  {\n"
    "    \"node_1\": {\"label\": ..., \"name\": ..., \"age\": ..., \"occupation\": ...},\n"
    "    \"node_2\": {\"label\": ..., \"name\": ..., ...},\n"
    "    \"relationship\": \"...\",\n"
    "    \"edge_properties\": {\"since\": ..., ...}\n"
    "  },\n"
    "]"
)

text = """John Smith, a 45-year-old software engineer from San Francisco, works at TechNova Inc., a leading AI research company founded in 2012. He has been leading the Natural Language Processing (NLP) team there since 2018.

John's wife, Emily Clark, is a 42-year-old physician currently working at St. Mary's Hospital. They got married in 2010 and have two children: Lily, aged 12, and Max, aged 9.

In 2021, John gave a keynote speech at the International AI Conference held in Berlin, where he presented a paper on ethical machine learning. He also collaborates with Professor Alan Cheng from Stanford University on explainable AI research.

During the COVID-19 pandemic in 2020, Emily volunteered in the ICU department for over six months, earning a community award for outstanding medical service.

They currently live in a townhouse near Golden Gate Park and frequently donate to the San Francisco Education Fund, supporting STEM education for underprivileged students."""

response = llm.generate(user_message=text, system_message=system_prompt)

# with open("split1.txt", "r", encoding="utf-8") as f:
#     text1 = f.read()
#
# response = llm.generate(user_message=text1, system_message=system_prompt)

# 提取 JSON 块
json_text = re.search(r"\[.*\]", response, re.DOTALL)
if not json_text:
    raise ValueError("No valid JSON found in LLM response")

edges = json.loads(json_text.group())

print(json.dumps(edges, indent=2, ensure_ascii=False))

# ✅ Save to JSON file
output_path = "output_knowledge_graph.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(edges, f, ensure_ascii=False, indent=2)

print(f"\nKnowledge graph saved to: {output_path}")