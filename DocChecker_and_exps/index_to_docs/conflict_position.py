"""
首先从原始文本中提取完整的三元组，然后在所有三元组中筛选与用户指定实体相关的三元组
对用户筛选的三元组，定位头实体和尾实体对应的上下文
"""
import os
from dotenv import load_dotenv
from src.kg_gen import KGGen
import tiktoken
import re
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# 配置参数
MAX_TOKENS = 512
STRIDE = 64

def find_entity_sentences(text, entity):
    """
    定位包含实体的完整句子（忽略大小写）
    返回所有包含该实体的句子列表
    """
    sentences = sent_tokenize(text)
    matched_sents = [s for s in sentences if re.search(re.escape(entity), s, re.I)]
    return matched_sents

def save_full_contexts_to_file(all_relations, full_text, filepath="entity_full_contexts.txt", max_sents=5):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, (h, r, t) in enumerate(all_relations, 1):
            f.write(f"Triple {i}: ({h}) -[{r}]-> ({t})\n\n")

            h_sents = find_entity_sentences(full_text, h)
            f.write(f"Head entity sentences for '{h}':\n")
            for sent in h_sents[:max_sents]:
                f.write(sent.strip() + "\n")
            f.write("\n")

            t_sents = find_entity_sentences(full_text, t)
            f.write(f"Tail entity sentences for '{t}':\n")
            for sent in t_sents[:max_sents]:
                f.write(sent.strip() + "\n")
            f.write("\n" + "="*60 + "\n\n")

def split_text_by_token_limit(text, max_tokens=MAX_TOKENS, stride=STRIDE, model_name="gpt-5.2"):
    """分块处理长文本"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - stride
    return chunks

def filter_triples(triples, target_entities):
    """
    筛选包含任意目标实体的三元组（不区分头尾位置）
    :param triples: 三元组列表 [(head, relation, tail)]
    :param target_entities: 目标实体列表 ["entity1", "entity2"]
    :return: 筛选后的三元组列表
    """
    filtered = []
    for h, r, t in triples:
        # 检查三元组中是否包含任意目标实体（不区分大小写）
        for entity in target_entities:
            if (entity.lower() in h.lower() or
                entity.lower() in t.lower()):
                filtered.append((h, r, t))
                break  # 只要匹配一个实体就满足条件
    return filtered

if __name__ == "__main__":
    load_dotenv()
    kg = KGGen(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL")
    )

    # 读取文本
    with open("book_chunk.txt", "r", encoding="utf-8") as file:
        full_text = file.read()

    # 分块处理
    chunks = split_text_by_token_limit(full_text)
    all_relations = []

    # 用户指定的目标实体列表（示例）
    target_entities = [
        "clerk",
        "Scrooge",
        "Merry Christmas"
    ]

    print(f"\nSearching for triples containing: {', '.join(target_entities)}")

    # 处理每个分块
    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")
        try:
            graph = kg.generate(input_data=chunk, model="gpt-5.2")
            if graph and hasattr(graph, 'relations'):
                # 应用筛选条件
                filtered = filter_triples(graph.relations, target_entities)
                if filtered:
                    all_relations.extend(filtered)
                    print(f"Found {len(filtered)} matching relations")
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            continue

    # 打印筛选结果
    if all_relations:
        print("\nMatching Triples:")
        for i, (h, r, t) in enumerate(all_relations, 1):
            # 高亮显示匹配的实体
            def highlight_entity(text):
                # for entity in target_entities:
                #     if entity.lower() in text.lower():
                #         return f"**{text}**"
                return text

            print(f"{i}. ({highlight_entity(h)}) -[{r}]-> ({highlight_entity(t)})")
        print(f"\nTotal matching triples found: {len(all_relations)}")
    else:
        print("\nNo triples contained the specified entities")

    if all_relations:
        save_full_contexts_to_file(all_relations, full_text, filepath="entity_full_contexts.txt")
        print(f"Full sentence contexts saved to 'entity_full_contexts.txt'")
    else:
        print("No triples contained the specified entities")