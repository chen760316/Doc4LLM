import os
import re
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from src.kg_gen import KGGen
import tiktoken

# 滑动窗口参数
MAX_TOKENS = 512
STRIDE = 64


def split_text_by_token_limit(text, max_tokens=512, stride=64, model_name="gpt-4o-mini"):
    """Split text into chunks with token limits and overlap."""
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


def find_entity_positions(text: str, entity: str) -> List[Tuple[int, int]]:
    """查找实体在文本中的所有出现位置"""
    return [(m.start(), m.end()) for m in re.finditer(re.escape(entity), text)]


if __name__ == "__main__":
    load_dotenv()

    try:
        kg = KGGen(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL")
        )
    except Exception as e:
        print(f"Failed to initialize KGGen: {e}")
        exit(1)

    try:
        with open("book_chunk.txt", "r", encoding="utf-8") as file:
            full_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

    chunks = split_text_by_token_limit(full_text, MAX_TOKENS, STRIDE)
    all_relations = []

    # 记录每个chunk在全文中的字符偏移量
    chunk_offsets = []
    current_offset = 0
    for chunk in chunks:
        chunk_offsets.append(current_offset)
        current_offset += len(chunk) - STRIDE * 4  # 近似计算重叠部分的偏移

    for i, (chunk, chunk_offset) in enumerate(zip(chunks, chunk_offsets)):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")

        try:
            graph = kg.generate(input_data=chunk, model="gpt-4o-mini")
            if not hasattr(graph, 'relations'):
                continue

            for head, relation, tail in graph.relations:
                # 在chunk中查找实体位置
                head_positions = [(chunk_offset + start, chunk_offset + end)
                                  for start, end in find_entity_positions(chunk, head)]
                tail_positions = [(chunk_offset + start, chunk_offset + end)
                                  for start, end in find_entity_positions(chunk, tail)]

                all_relations.append({
                    "head": {"text": head, "positions": head_positions},
                    "relation": relation,
                    "tail": {"text": tail, "positions": tail_positions}
                })

            print(f"Extracted {len(graph.relations)} relations from this chunk")
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")

    # 打印结果
    print("\n=== 知识图谱三元组（带位置信息） ===")
    for i, triple in enumerate(all_relations, 1):
        head = triple["head"]
        tail = triple["tail"]
        print(f"\n{i}. {head['text']} [位置: {head['positions']}]")
        print(f"   → {triple['relation']} →")
        print(f"   {tail['text']} [位置: {tail['positions']}]")

        # 打印上下文示例（第一个找到的位置）
        if head['positions']:
            start, end = head['positions'][0]
            context = full_text[max(0, start - 20):min(len(full_text), end + 20)]
            print(f"   头实体上下文: ...{context}...")
        if tail['positions']:
            start, end = tail['positions'][0]
            context = full_text[max(0, start - 20):min(len(full_text), end + 20)]
            print(f"   尾实体上下文: ...{context}...")

    print(f"\n处理完成！共提取 {len(all_relations)} 个三元组")