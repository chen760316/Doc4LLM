"""
在v2的基础上记录节点和关系在原始文本中的索引
"""

from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
import difflib

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

def find_text_span(text, substring):
    """
    找到 substring 在 text 中的第一个出现位置，返回 (start, end) 索引。
    如果找不到，返回 None。
    """
    match = re.search(re.escape(substring), text)
    if match:
        return (match.start(), match.end())
    else:
        return None

def add_text_spans_to_graph(edges, original_text):
    """
    给每个节点和边添加 text_span 属性，表示它们在原文中的位置。
    """
    for edge in edges:
        # 节点1位置
        name1 = edge["node_1"].get("name", "")
        span1 = find_text_span(original_text, name1)
        if span1:
            edge["node_1"]["text_span"] = {"start": span1[0], "end": span1[1]}

        # 节点2位置
        name2 = edge["node_2"].get("name", "")
        span2 = find_text_span(original_text, name2)
        if span2:
            edge["node_2"]["text_span"] = {"start": span2[0], "end": span2[1]}

        # 关系描述位置
        relation = edge.get("relationship", "")
        span_rel = find_text_span(original_text, relation)
        if span_rel:
            if "edge_properties" not in edge:
                edge["edge_properties"] = {}
            edge["edge_properties"]["text_span"] = {"start": span_rel[0], "end": span_rel[1]}

    return edges

# 模糊匹配函数，找出文本中与关系描述最相似的片段位置
def find_fuzzy_match_span(text, pattern, window=100):
    text_len = len(text)
    pattern_len = len(pattern)
    best_ratio = 0
    best_span = (0, 0)

    if pattern_len >= text_len:
        return (0, text_len)

    step = max(10, pattern_len // 2)
    search_window = max(pattern_len, window)

    for start in range(0, text_len, step):
        end = min(start + search_window, text_len)
        segment = text[start:end]
        ratio = difflib.SequenceMatcher(None, segment, pattern).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_span = (start, end)
        if best_ratio > 0.95:
            break

    return best_span

# 为每个节点和关系标注文本位置
def add_text_spans_with_fuzzy_relation(edges, original_text):
    for edge in edges:
        # 记录node_1的name位置
        name1 = edge["node_1"].get("name", "")
        if name1:
            start = original_text.find(name1)
            if start != -1:
                edge["node_1"]["text_span"] = {"start": start, "end": start + len(name1)}

        # 记录node_2的name位置
        name2 = edge["node_2"].get("name", "")
        if name2:
            start = original_text.find(name2)
            if start != -1:
                edge["node_2"]["text_span"] = {"start": start, "end": start + len(name2)}

        # 模糊匹配关系描述位置
        relation = edge.get("relationship", "")
        if relation:
            span = find_fuzzy_match_span(original_text, relation)
            if span:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["text_span"] = {"start": span[0], "end": span[1]}

    return edges

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
# text = """John Smith, a 45-year-old software engineer from San Francisco, works at TechNova Inc., a leading AI research company founded in 2012. He has been leading the Natural Language Processing (NLP) team there since 2018.
#
# John's wife, Emily Clark, is a 42-year-old physician currently working at St. Mary's Hospital. They got married in 2010 and have two children: Lily, aged 12, and Max, aged 9.
#
# In 2021, John gave a keynote speech at the International AI Conference held in Berlin, where he presented a paper on ethical machine learning. He also collaborates with Professor Alan Cheng from Stanford University on explainable AI research.
#
# During the COVID-19 pandemic in 2020, Emily volunteered in the ICU department for over six months, earning a community award for outstanding medical service.
#
# They currently live in a townhouse near Golden Gate Park and frequently donate to the San Francisco Education Fund, supporting STEM education for underprivileged students."""
#
# response = llm.generate(user_message=text, system_message=system_prompt)

with open("text1.txt", "r", encoding="utf-8") as f:
    text = f.read()

response = llm.generate(user_message=text, system_message=system_prompt)

# 提取 JSON 块
json_text = re.search(r"\[.*\]", response, re.DOTALL)
if not json_text:
    raise ValueError("No valid JSON found in LLM response")

edges = json.loads(json_text.group())

# 添加索引信息
edges_with_spans = add_text_spans_with_fuzzy_relation(edges, text)

# 打印格式化结果
print(json.dumps(edges_with_spans, indent=2, ensure_ascii=False))

# ✅ Save to JSON file
output_path = "graph1.json"
# 保存到文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(edges_with_spans, f, ensure_ascii=False, indent=2)

print(f"\nKnowledge graph saved to: {output_path}")