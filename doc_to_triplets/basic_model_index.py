"""
只能定位部分三元组的位置
"""

from dotenv import load_dotenv
import os
from src.kg_gen import KGGen
from transformers import AutoTokenizer

# 你之前的滑动窗口参数和代码
MAX_TOKENS = 384
STRIDE = 64

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 加入 fuzzy_find
import difflib

def fuzzy_find(substring, text, threshold=0.8):
    if not substring or not text:
        return -1

    len_sub = len(substring)
    best_ratio = 0
    best_start = -1

    for i in range(len(text) - len_sub + 1):
        window = text[i:i + len_sub + 5]
        ratio = difflib.SequenceMatcher(None, window.lower(), substring.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_start = i

    if best_ratio >= threshold:
        return best_start
    else:
        return -1


def chunk_text_by_tokens(text, tokenizer, max_tokens=MAX_TOKENS, stride=STRIDE):
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    tokens = encoding.tokens()

    token_chunks = []
    start_token_idx = 0
    while start_token_idx < len(tokens):
        end_token_idx = min(start_token_idx + max_tokens, len(tokens))

        # 原始字符级起止位置
        start_char = offsets[start_token_idx][0]
        end_char = offsets[end_token_idx - 1][1]

        original_chunk_text = text[start_char:end_char]

        token_chunks.append({
            "text": original_chunk_text,
            "start_char": start_char
        })

        start_token_idx += max_tokens - stride

    return token_chunks



if __name__ == "__main__":
    load_dotenv()

    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))

    # 这里假设你已经有一个 tokenizer 实例，比如 transformers 的 tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 示例，请根据实际替换

    # 你的全文本
    with open("book_split.txt", "r", encoding="utf-8") as f:
        full_text = f.read()

    # 按滑动窗口切分文本
    chunks = chunk_text_by_tokens(full_text, tokenizer)

    # 汇总所有片段抽取的三元组
    all_graphs = []
    for chunk in chunks:
        graph = kg.generate(input_data=chunk, model="openai/gpt-4o-mini")
        all_graphs.append(graph)

    # 你可以根据 kg.generate 返回的具体数据结构，汇总或合并这些图谱
    # 这里简单假设 graph.relations 是三元组列表
    # 汇总所有片段抽取的三元组
    all_triples = []
    for chunk in chunks:
        graph = kg.generate(input_data=chunk["text"], model="openai/gpt-4o-mini")

        if hasattr(graph, "relations"):
            for head, relation, tail in graph.relations:
                # 在 chunk["text"]（原始 text 切片）中查找
                head_pos = fuzzy_find(head, chunk["text"])
                tail_pos = fuzzy_find(tail, chunk["text"])

                if head_pos != -1:
                    head_abs = chunk["start_char"] + head_pos
                else:
                    head_abs = None

                if tail_pos != -1:
                    tail_abs = chunk["start_char"] + tail_pos
                else:
                    tail_abs = None

                all_triples.append((head, relation, tail, head_abs, tail_abs))


    # 输出所有抽取的三元组及其位置信息
    for head, relation, tail, head_pos, tail_pos in all_triples:
        print(f"({head}) -[{relation}]-> ({tail})", end="")
        if head_pos is not None and tail_pos is not None:
            print(f" [head_pos: {head_pos}, tail_pos: {tail_pos}]")
        else:
            print(" [position not found]")

    # 也可以生成可视化文件（合并后的全部三元组）
    # visualize(all_graphs, "E:/Document cleaning/kg-gen/tests/test_basic.html")  # 这里你可能需要先合并graph结构
