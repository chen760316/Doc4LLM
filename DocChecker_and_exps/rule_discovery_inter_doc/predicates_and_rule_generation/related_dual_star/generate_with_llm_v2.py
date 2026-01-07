"""
采用更灵活的prompts
无需指定固定的返回格式，补全顶点和边所需的所有属性
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

system_prompt = (
    "You are an expert in extracting knowledge graphs from natural language text.\n"
    "Given an input text, extract all entities and relationships mentioned.\n"
    "For each entity, include all relevant attributes found in the text—such as age, occupation, nationality, status, location, etc.\n"
    "For each relationship, include a natural language description and, if available, any attributes such as time, place, reason, or method.\n"
    "Return your output as a JSON array, following this flexible schema:\n"
    "[\n"
    "  {\n"
    '    "node_1": { "label": "<ontology_label>", "name": "<entity_name>", "...": "<other attributes if available>" },\n'
    '    "node_2": { "label": "<ontology_label>", "name": "<entity_name>", "...": "<other attributes if available>" },\n'
    '    "relationship": "<natural language description of the relation>",\n'
    '    "edge_properties": { "...": "<additional relationship attributes if available>" }\n'
    "  }\n"
    "]\n"
    "Respond with only the JSON array—no explanations, no markdown, no extra text."
)

# 文本一
with open("../text2.txt", "r", encoding="utf-8") as file:
    text = file.read()

response = llm.generate(user_message=text, system_message=system_prompt)

# 文本二
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
output_path = "output_knowledge_graph_text2.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(edges, f, ensure_ascii=False, indent=2)

print(f"\nKnowledge graph saved to: {output_path}")