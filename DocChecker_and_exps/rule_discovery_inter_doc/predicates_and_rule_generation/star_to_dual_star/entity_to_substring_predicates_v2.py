"""
生成顶点和原始文档text1.txt和text2.txt中的子字符串的对应关系
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
    with open("../text2.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # 加载知识图谱
    with open("output_knowledge_graph_text2.json", "r", encoding="utf-8") as f:
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
    with open("entity_text2.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

