"""
用于定位文本中的三元组和冲突位置
用于对三元组引入的冲突在原文进行修复
制定一种策略来标记每份文档中的错误，同时标注出与其他句子、章节或文档不一致之处。如果文件𝐷中的某条陈述与多处声明相矛盾，进行处理
"""
import os
from dotenv import load_dotenv
from src.kg_gen import KGGen
import tiktoken
import re
from typing import List, Tuple, Optional, Dict
from difflib import SequenceMatcher
from openai import OpenAI
import ast
from collections import defaultdict

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

def suggest_corrections_from_conflicts(full_text: str, triple_positions: dict, conflict_map: Dict[int, List[int]], all_triples: List[Tuple[str, str, str]]) -> str:
    """
    对存在冲突的三元组进行文本定位和修复，输出带标注的文档：
    - 显式标注出与其他句子/段落冲突的内容；
    - 提供修改建议；
    - 如果同一个三元组与多个句子冲突，将这些句子列出统一处理；
    - 尽可能一次性处理所有冲突片段；
    """
    corrected_text = full_text
    offset = 0  # 修正位置偏移量

    processed_indices = set()  # 防止重复处理

    for idx, conflict_indices in conflict_map.items():
        if idx in processed_indices:
            continue

        # 收集所有相关三元组的 index
        group_indices = set([idx] + conflict_indices)
        processed_indices.update(group_indices)

        # 收集所有冲突文本片段
        segments = []
        triple_strs = []

        for i in group_indices:
            triple = all_triples[i]
            triple_strs.append(f"({triple[0]}) -[{triple[1]}]-> ({triple[2]})")

            position = triple_positions.get(i, {}).get("position")
            if not position:
                continue

            h_start, h_end, _, _, t_start, t_end = position
            start = h_start
            end = t_end

            segment = corrected_text[start + offset:end + offset]
            segments.append(segment.strip())

        # 构造 prompt
        triple_list_text = "\n".join(f"- {t}" for t in triple_strs)
        segment_text = "\n".join(f"- \"{s}\"" for s in segments)

        prompt = f"""
You are a logical editor working on resolving textual contradictions.

The following sentences contain conflicting or inconsistent statements:

📌 Conflicting Sentences:
{segment_text}

🧩 Related Triples:
{triple_list_text}

👉 Task:
Unify the meaning across these conflicting statements by suggesting a revised version that resolves the contradictions. If appropriate, explain a change in character, timeline, or mood that justifies the differences.

Return in the format:
- [REVISED TEXT]: ...
- [EXPLANATION]: (optional)
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()

            # 将标注内容插入到所有相关位置后面（只插一次）
            insert_position = max(t_end for i in group_indices if (triple_positions.get(i, {}).get("position")) for _, _, _, _, _, t_end in [triple_positions[i]['position']])
            annotated = f"\n\n🔍 [CONFLICT GROUP DETECTED]\n{result}\n\n"

            corrected_text = (
                corrected_text[:insert_position + offset] +
                annotated +
                corrected_text[insert_position + offset:]
            )
            offset += len(annotated)

        except Exception as e:
            print(f"⚠️ Error resolving conflict group {triple_strs}: {e}")
            continue

    return corrected_text


def parse_conflict_analysis(analysis_blocks: List[str], all_triples: List[Tuple[str, str, str]]) -> Dict[int, List[int]]:
    """
    将 GPT 返回的冲突分析文本转为结构化映射：
    返回 conflict_map: {idx -> [conflicting idxs]}
    """
    triple_to_index = {f"({h}) -[{r}]-> ({t})": idx for idx, (h, r, t) in enumerate(all_triples)}
    conflict_map = defaultdict(list)

    for block in analysis_blocks:
        try:
            # 使用正则提取 `Conflicting Triples: [...]` 行
            match = re.search(r"Conflicting Triples:\s*(\[[\s\S]*?\])", block)
            if not match:
                continue

            conflicting_list_str = match.group(1)

            # 转换为列表
            conflicting_list = ast.literal_eval(conflicting_list_str)

            # 找到这些三元组在 all_triples 中的 index
            conflicting_indices = [triple_to_index[triple] for triple in conflicting_list if triple in triple_to_index]

            # 构建互相映射
            for i in conflicting_indices:
                for j in conflicting_indices:
                    if i != j and j not in conflict_map[i]:
                        conflict_map[i].append(j)

        except Exception as e:
            print(f"⚠️ Error parsing conflict block: {e}")
            continue

    return dict(conflict_map)


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

def build_conflict_map_from_structured_analysis(structured_conflict_json: List[dict], all_triples: List[Tuple[str, str, str]]) -> Dict[int, List[int]]:
    conflict_map = {}
    triple_to_index = {t: idx for idx, t in enumerate(all_triples)}

    for analysis in structured_conflict_json:
        conflict_triples = analysis.get("Conflicting Triples", [])
        group_indices = []

        for t_str in conflict_triples:
            t_str = t_str.strip("()")
            parts = t_str.split(" -[")
            if len(parts) != 2 or "]-> " not in parts[1]:
                continue
            head = parts[0].strip()
            relation, tail = parts[1].split("]-> ")
            relation = relation.strip()
            tail = tail.strip("()").strip()
            triple = (head, relation, tail)
            idx = triple_to_index.get(triple)
            if idx is not None:
                group_indices.append(idx)

        # 建立映射
        for i in group_indices:
            conflict_map[i] = [j for j in group_indices if j != i]

    return conflict_map


if __name__ == "__main__":
    load_dotenv()

    kg = KGGen(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL")
    )

    with open("book_chunk_conflict_correction.txt", "r", encoding="utf-8") as file:
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

    # --- 添加在 conflict report 写入之后 ---
    with open("conflict_from_kg_report.txt", "w", encoding="utf-8") as out_file:
        out_file.write("\n\n".join(final_results))
    print("✅ All chunks processed. Conflict report saved to 'conflict_from_kg_report.txt'")

    # ➕ [添加这段]
    # 解析结构化冲突映射
    conflict_map = parse_conflict_analysis(final_results, all_relations)
    print(f"🔍 Parsed structured conflict map with {len(conflict_map)} conflicted triples.")

    corrected = suggest_corrections_from_conflicts(
        full_text, triple_positions, conflict_map, all_relations
    )

    with open("book_chunk_conflict_correction.txt", "w", encoding="utf-8") as f:
        f.write(corrected)

    print("✅ Final corrected document saved to 'book_chunk_conflict_correction.txt'")