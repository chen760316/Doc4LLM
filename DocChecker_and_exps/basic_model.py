from dotenv import load_dotenv
import os
from src.kg_gen import KGGen
from transformers import AutoTokenizer

# 你之前的滑动窗口参数和代码
MAX_TOKENS = 384
STRIDE = 64

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def chunk_text_by_tokens(text, tokenizer, max_tokens=MAX_TOKENS, stride=STRIDE):
    tokens = tokenizer.tokenize(text)
    token_chunks = []
    start_token_idx = 0
    while start_token_idx < len(tokens):
        end_token_idx = min(start_token_idx + max_tokens, len(tokens))
        chunk_tokens = tokens[start_token_idx:end_token_idx]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        token_chunks.append(chunk_text)
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
    all_triples = []
    for graph in all_graphs:
        if hasattr(graph, "relations"):
            all_triples.extend(graph.relations)

    # 输出所有抽取的三元组
    for head, relation, tail in all_triples:
        print(f"({head}) -[{relation}]-> ({tail})")

    # 也可以生成可视化文件（合并后的全部三元组）
    # visualize(all_graphs, "E:/Document cleaning/kg-gen/tests/test_basic.html")  # 这里你可能需要先合并graph结构
