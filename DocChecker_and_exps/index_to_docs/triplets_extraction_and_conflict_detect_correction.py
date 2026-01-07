"""
用于定位文本中的三元组和冲突位置
用于对三元组引入的冲突在原文进行修复
"""
import os
from dotenv import load_dotenv
from src.kg_gen import KGGen
import tiktoken
import re
from typing import List, Tuple, Optional
from difflib import SequenceMatcher
from openai import OpenAI

# 滑动窗口参数
MAX_TOKENS = 512
STRIDE = 64

# 初始化 OpenAI 客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)


class TextPositionFinder:
    @staticmethod
    def similar(a: str, b: str) -> float:
        """计算两个字符串的相似度"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    @staticmethod
    def find_approximate_positions(text: str, pattern: str, threshold: float = 0.6) -> List[Tuple[int, int]]:
        """
        使用模糊匹配查找文本中与模式相似的位置
        :param text: 原始文本
        :param pattern: 要查找的模式
        :param threshold: 相似度阈值
        :return: 匹配位置的(start, end)列表
        """
        positions = []
        pattern_lower = pattern.lower()
        text_lower = text.lower()
        pattern_len = len(pattern)

        if pattern_len < 2:
            return []

        # 使用正则表达式进行模糊匹配
        regex_pattern = re.escape(pattern_lower)
        regex_pattern = regex_pattern.replace(r"\ ", r"\s+")  # 允许空格变化

        try:
            for match in re.finditer(regex_pattern, text_lower):
                start, end = match.span()
                matched_text = text[start:end]
                similarity = TextPositionFinder.similar(matched_text, pattern)
                if similarity >= threshold:
                    positions.append((start, end))
        except:
            # 回退到简单匹配
            window_size = max(pattern_len, 3)
            for i in range(len(text) - window_size + 1):
                window = text[i:i + window_size]
                similarity = TextPositionFinder.similar(window, pattern)
                if similarity >= threshold:
                    positions.append((i, i + window_size))

        # 合并重叠的位置
        if not positions:
            return []

        merged = [positions[0]]
        for current in positions[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

        return merged

    @staticmethod
    def find_triple_positions(text: str, head: str, relation: str, tail: str) -> List[Tuple[int, int, Optional[int], Optional[int], int, int]]:
        """
        查找三元组位置，允许关系缺失
        :return: [(head_start, head_end, rel_start or None, rel_end or None, tail_start, tail_end)]
        """
        head_pos = TextPositionFinder.find_approximate_positions(text, head)
        rel_pos = TextPositionFinder.find_approximate_positions(text, relation)
        tail_pos = TextPositionFinder.find_approximate_positions(text, tail)

        results = []

        if not head_pos or not tail_pos:
            return []

        # 尝试所有头尾组合
        for h_start, h_end in head_pos[:3]:
            for t_start, t_end in tail_pos[:3]:
                if t_start < h_end:  # 尾实体不能在头实体之前
                    continue

                # 尝试找到关系
                found_rel = False
                if rel_pos:
                    for r_start, r_end in rel_pos:
                        if h_end <= r_start and r_end <= t_start:
                            results.append((h_start, h_end, r_start, r_end, t_start, t_end))
                            found_rel = True

                # 即使没找到关系也记录
                if not found_rel:
                    results.append((h_start, h_end, None, None, t_start, t_end))

        return results


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

def ask_conflict_about_triples(text_chunk, triples):
    """向 GPT-4o 查询：这些三元组是否存在冲突或模糊性"""
    triple_text = "\n".join([f"({h}) -[{r}]-> ({t})" for h, r, t in triples])
    prompt = f"""
You are a knowledge graph and text reasoning expert.

Given the following text chunk and the extracted triples, identify:
1. If any conflicting statements or logical contradictions exist among the triples;
2. If any triple or part of it is vague, speculative, or semantically ambiguous.

Return in structured format:
- Conflicting Triples: [...]
- Uncertain/Vague Triples: [...]
- Entities Involved in Conflict: [...]

Text:
\"\"\"
{text_chunk}
\"\"\"

Triples:
{triple_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error querying GPT-4o: {e}]"

def suggest_corrections_from_conflicts(full_text: str, triple_positions: dict, conflict_analysis: List[str]) -> str:
    """
    精细化修复：仅替换与冲突三元组直接相关的最小文本片段（头、尾、关系之间区域）。
    保留上下文句式风格，避免大段替换。
    """
    corrected_text = full_text
    offset = 0  # 用于修正位置偏移

    for idx, data in triple_positions.items():
        triple = data["triple"]
        position = data["position"]

        if not position:
            continue

        h_start, h_end, r_start, r_end, t_start, t_end = position
        triple_str = f"({triple[0]}) -[{triple[1]}]-> ({triple[2]})"

        # 判断是否涉及冲突
        if not any(triple_str in section for section in conflict_analysis):
            continue

        # 精准提取冲突区域文本（只提取包含头、关系、尾之间的部分）
        focus_start = h_start
        focus_end = t_end
        context = corrected_text[focus_start + offset:focus_end + offset]

        # 构造更小的 prompt，只对焦点语义修改
        prompt = f"""
You are an expert in logical text editing.

The following short text segment contains a factual inconsistency based on the triple:
Triple: {triple_str}

Please revise this **only if necessary** to resolve the logical inconsistency,
and preserve the original phrasing, tone, and structure as much as possible.

Text:
\"\"\"
{context}
\"\"\"

Return only the corrected version. If no change is needed, return the original text.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            revised = response.choices[0].message.content.strip()

            if revised != context:
                print(f"🔁 Replacing: {context} → {revised}")
                corrected_text = (
                    corrected_text[:focus_start + offset] +
                    revised +
                    corrected_text[focus_end + offset:]
                )
                offset += len(revised) - (focus_end - focus_start)

        except Exception as e:
            print(f"⚠️ Error correcting triple {triple_str}: {e}")
            continue

    return corrected_text



def locate_triples_in_full_text(full_text: str, triples: list) -> dict:
    """
    在全文中定位所有三元组的位置，每个三元组只保留最接近的一个匹配
    :return: {index: {'triple': (h,r,t), 'position': (h_start,h_end,r_start,r_end,t_start,t_end)}}
    """
    results = {}
    for idx, (head, relation, tail) in enumerate(triples):
        positions = TextPositionFinder.find_triple_positions(full_text, head, relation, tail)

        if not positions:
            results[idx] = {
                'triple': (head, relation, tail),
                'position': None
            }
            continue

        # 选择最接近的匹配：优先选择关系存在的，然后选择头尾距离最近的
        best_match = None
        min_distance = float('inf')

        for pos in positions:
            h_start, h_end, r_start, r_end, t_start, t_end = pos

            # 如果关系存在，优先考虑
            if r_start is not None:
                if best_match is None or (r_start - h_end) + (t_start - r_end) < min_distance:
                    best_match = pos
                    min_distance = (r_start - h_end) + (t_start - r_end)
            else:
                # 关系不存在，只考虑头尾距离
                current_distance = t_start - h_end
                if best_match is None or current_distance < min_distance:
                    best_match = pos
                    min_distance = current_distance

        results[idx] = {
            'triple': (head, relation, tail),
            'position': best_match
        }

    return results


if __name__ == "__main__":
    load_dotenv()

    kg = KGGen(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL")
    )

    with open("book_chunk_conflict.txt", "r", encoding="utf-8") as file:
        full_text = file.read()

    # Split text into chunks
    chunks = split_text_by_token_limit(full_text, max_tokens=MAX_TOKENS, stride=STRIDE)

    all_relations = []
    final_results = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")
        try:
            graph = kg.generate(input_data=chunk, model="gpt-4")
            if graph and hasattr(graph, 'relations'):
                all_relations.extend(graph.relations)
                triples = graph.relations
                print(f"✅ Extracted {len(triples)} triples. Now analyzing conflicts...")
                print(triples)
                analysis = ask_conflict_about_triples(full_text, triples)
                print(f"\n📘 Conflict Analysis for Chunk {i + 1}:\n{analysis}\n{'-'*80}\n")
                final_results.append(f"Chunk {i+1}:\n{analysis}")
                print(f"Extracted {len(graph.relations)} relations")
            else:
                print("⚠️ No triples extracted.")
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            continue

    # 定位所有三元组在原文中的位置
    triple_positions = locate_triples_in_full_text(full_text, all_relations)
    with open("conflict_from_kg_report.txt", "w", encoding="utf-8") as out_file:
        out_file.write("\n\n".join(final_results))
    print("✅ All chunks processed. Conflict report saved to 'conflict_from_kg_report.txt'")

    # 打印结果
    print("\nFinal Extracted Triples with Best Positions:")
    for idx, data in triple_positions.items():
        print("*" * 100)
        head, relation, tail = data['triple']
        position = data['position']

        print(f"\n{idx + 1}. Triple: ({head}) -[{relation}]-> ({tail})")

        if not position:
            print("  No matching head or tail entities found")
            continue

        h_start, h_end, r_start, r_end, t_start, t_end = position

        print(f"  Best Match:")
        print(f"    Head[{h_start}:{h_end}]: {full_text[h_start:h_end]}")

        if r_start is not None and r_end is not None:
            print(f"    Relation[{r_start}:{r_end}]: {full_text[r_start:r_end]}")
        else:
            print("    Relation: NOT FOUND")

        print(f"    Tail[{t_start}:{t_end}]: {full_text[t_start:t_end]}")

        # 打印上下文
        # context_start = max(0, min(h_start, t_start) - 20)
        # context_end = min(len(full_text), max(h_end, t_end) + 20)
        # print(f"    Context: ...{full_text[context_start:context_end]}...")

    print(f"\nTotal triples extracted: {len(all_relations)}")
    print(f"Total triples with matches: {sum(1 for x in triple_positions.values() if x['position'])}")

    corrected = suggest_corrections_from_conflicts(full_text, triple_positions, final_results)

    with open("book_chunk_conflict_corrected.txt", "w", encoding="utf-8") as f:
        f.write(corrected)

    print("✅ Final corrected document saved to 'book_chunk_conflict_corrected.txt'")