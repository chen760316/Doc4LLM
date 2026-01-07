"""
实现谓词𝑆(𝑥, 𝑠)
𝑠是一个字符串变量，用于文档中的子字符串
𝑆(𝑥, 𝑠)将原始文档中描述实体𝑥的子字符串𝑠与实体𝑥绑定，而不是使用大模型生成实体𝑥的自然语言描述
生成节点x和原始文档中子字符串s的一一对应关系triple_entity_spans.json
useful
"""

import json
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')


def find_best_matching_span(entity_dict, text_sentences):
    """
    给定实体属性字典，在原始文本句子中寻找最相关的句子片段。
    """
    name = entity_dict.get("name", "")
    name_parts = name.split()
    aliases = set([name] + name_parts)

    # 构造关键词集合：名字 + 属性值（字符串）
    keywords = set(aliases)
    for key, value in entity_dict.items():
        if isinstance(value, str):
            keywords.add(value)

    # 尝试在原文中匹配这些关键词，选包含关键词最多的句子或相邻句组
    max_score = 0
    best_span = ""

    for i in range(len(text_sentences)):
        sent = text_sentences[i]
        score = sum(1 for kw in keywords if kw in sent)

        # 可选：合并下一句一起考虑上下文
        if i + 1 < len(text_sentences):
            sent_pair = sent + " " + text_sentences[i + 1]
            score_pair = sum(1 for kw in keywords if kw in sent_pair)
            if score_pair > max_score:
                max_score = score_pair
                best_span = sent_pair
        if score > max_score:
            max_score = score
            best_span = sent

    return best_span


def extract_entity_spans_per_triple(text, triples):
    sentences = sent_tokenize(text)
    enriched_triples = []

    for triple in triples:
        span1 = find_best_matching_span(triple["node_1"], sentences)
        span2 = find_best_matching_span(triple["node_2"], sentences)

        enriched_triples.append({
            "triple": triple,
            "spans": {
                "node_1_span": span1,
                "node_2_span": span2
            }
        })

    return enriched_triples


# === 示例运行 ===
if __name__ == "__main__":
    # 原始文本
    text = """John Smith, a 45-year-old software engineer from San Francisco, works at TechNova Inc., a leading AI research company founded in 2012. He has been leading the Natural Language Processing (NLP) team there since 2018.

    John's wife, Emily Clark, is a 42-year-old physician currently working at St. Mary's Hospital. They got married in 2010 and have two children: Lily, aged 12, and Max, aged 9.

    In 2021, John gave a keynote speech at the International AI Conference held in Berlin, where he presented a paper on ethical machine learning. He also collaborates with Professor Alan Cheng from Stanford University on explainable AI research.

    During the COVID-19 pandemic in 2020, Emily volunteered in the ICU department for over six months, earning a community award for outstanding medical service.

    They currently live in a townhouse near Golden Gate Park and frequently donate to the San Francisco Education Fund, supporting STEM education for underprivileged students."""

    # 加载知识图谱
    with open("../output_knowledge_graph.json", "r", encoding="utf-8") as f:
        triples = json.load(f)

    # 执行三元组实体片段提取
    result = extract_entity_spans_per_triple(text, triples)

    # 输出预览
    for item in result[:3]:  # 只看前3个示例
        print("🔹 Triple:", item["triple"]["relationship"])
        print(" - node_1_span:", item["spans"]["node_1_span"])
        print(" - node_2_span:", item["spans"]["node_2_span"])
        print()

    # 保存结果
    with open("triple_entity_spans.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

