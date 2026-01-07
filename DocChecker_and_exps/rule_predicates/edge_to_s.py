"""
实现谓词𝑆(e𝑥, 𝑠)，将关系（边）对应为原始文本中的自然语言
𝑠是一个字符串变量，用于文档中的子字符串
(a)𝑆(e𝑥, 𝑠)将子字符串𝑠与实体关系e𝑥绑定，这里的s对应原始文档中的上下文
生成边e𝑥和原始文档子字符串𝑠的一一对应关系保存进triple_relation_spans.json文件
useful
"""


import json
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
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

def get_keywords_from_triple(triple):
    """
    从 relationship 字符串和 edge_properties 中提取关键词集合
    """
    keywords = set()

    # 加入 relationship 字符串中的关键词（拆词处理）
    relationship_text = triple.get("relationship", "")
    for word in relationship_text.split():
        if len(word) > 2:
            keywords.add(word.strip(".,'\""))

    # 加入边属性中的字符串值
    edge_props = triple.get("edge_properties", {})
    for v in edge_props.values():
        if isinstance(v, str):
            for word in v.split():
                keywords.add(word.strip(".,'\""))
        elif isinstance(v, int):  # 年份或数字
            keywords.add(str(v))

    return keywords


def find_relation_context_span(triple, sentences):
    """
    在原始文本中为每个三元组选择最佳的上下文子句 span
    """
    keywords = get_keywords_from_triple(triple)

    best_score = 0
    best_span = ""

    for i in range(len(sentences)):
        window = " ".join(sentences[i:i + 2])  # 以两个句子的窗口滑动匹配
        score = sum(1 for kw in keywords if kw in window)

        if score > best_score:
            best_score = score
            best_span = window

    return best_span


def extract_relation_context_spans_per_triple(text, triples):
    """
    为每个三元组三元组添加精确的关系上下文子字符串
    """
    sentences = sent_tokenize(text)
    results = []

    for triple in triples:
        context_span = find_relation_context_span(triple, sentences)

        results.append({
            "triple": triple,
            "relation_context_span": context_span
        })

    return results

def extract_context_with_llm(text, triple, llm):
    """
    调用 LLM 提取与给定三元组关系最相关的上下文。
    """
    prompt = f"""
You are a helpful assistant. Given the following text and a relationship triple, please find the shortest
meaningful excerpt from the text that clearly describes the relationship.

Text:
\"\"\"{text}\"\"\"

Relationship:
\"\"\"{triple['relationship']}\"\"\"

Edge properties:
\"\"\"{json.dumps(triple.get('edge_properties', {}), ensure_ascii=False)}\"\"\"

Please output the excerpt exactly as it appears in the text, without adding or changing words.
If the exact excerpt cannot be found, provide the closest meaningful sentence(s).
Return ONLY the excerpt.
"""

    try:
        response = llm.generate(
            user_message=prompt,
            system_message="You are a precise extractor of text spans related to entity relationships."
        )
        return response.strip()
    except Exception as e:
        print(f"⚠️ LLM extraction error for triple {triple['relationship']}: {e}")
        return ""

def extract_relation_contexts_with_llm(text, triples):
    results = []
    for triple in triples:
        context_span = extract_context_with_llm(text, triple, llm)
        results.append({
            "triple": triple,
            "relation_context_span": context_span
        })
    return results


if __name__ == "__main__":
    with open("../output_knowledge_graph.json", "r", encoding="utf-8") as f:
        triples = json.load(f)

    text = """John Smith, a 45-year-old software engineer from San Francisco, works at TechNova Inc., a leading AI research company founded in 2012. He has been leading the Natural Language Processing (NLP) team there since 2018.

John's wife, Emily Clark, is a 42-year-old physician currently working at St. Mary's Hospital. They got married in 2010 and have two children: Lily, aged 12, and Max, aged 9.

In 2021, John gave a keynote speech at the International AI Conference held in Berlin, where he presented a paper on ethical machine learning. He also collaborates with Professor Alan Cheng from Stanford University on explainable AI research.

During the COVID-19 pandemic in 2020, Emily volunteered in the ICU department for over six months, earning a community award for outstanding medical service.

They currently live in a townhouse near Golden Gate Park and frequently donate to the San Francisco Education Fund, supporting STEM education for underprivileged students."""

    enriched = extract_relation_contexts_with_llm(text, triples)

    for item in enriched[:3]:
        print("Relationship:", item["triple"]["relationship"])
        print("Context span:", item["relation_context_span"])
        print()

    with open("triple_relation_spans.json", "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
