import os
import json
import re
import difflib
import time
from typing import Dict, Any, Optional, List, Tuple

# --- 配置 ---
MAX_RETRIES = 3 # 最大重试次数
INPUT_FILE = "dense_text.json" # 输入文件路径

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 初始化 LLM
from knowledge_graph_maker import OpenAIClient
llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9
)

# --- 辅助函数 (保持不变) ---

def find_text_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    """
    找到 substring 在 text 中的第一个出现位置，返回 (start, end) 索引。
    如果找不到，返回 None。
    """
    # 使用 re.escape 确保特殊字符被正确匹配
    match = re.search(re.escape(substring), text)
    if match:
        return (match.start(), match.end())
    return None

def find_fuzzy_match_span(text: str, pattern: str, window: int = 100) -> Optional[Tuple[int, int]]:
    """
    使用 difflib 进行模糊匹配，找到 pattern 在 text 中的最佳匹配范围。
    """
    text_len = len(text)
    pattern_len = len(pattern)
    best_ratio = 0.9  # 设置一个合理的阈值，防止匹配到无关内容
    best_span = None

    if pattern_len == 0 or pattern_len > text_len:
        return None

    # 遍历文本，寻找最佳匹配
    for i in range(text_len - pattern_len + 1):
        segment = text[i:i + pattern_len]
        ratio = difflib.SequenceMatcher(None, segment, pattern).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_span = (i, i + pattern_len)

    # 如果找到高相似度的匹配，返回其索引
    return best_span


def add_text_spans_with_fuzzy_relation(edges: List[Dict[str, Any]], original_text: str) -> List[Dict[str, Any]]:
    """
    遍历提取的边，利用模糊匹配和精确匹配为节点和关系添加 text_span 索引。
    """
    for edge in edges:
        # 节点 1 的索引 (精确匹配)
        name1 = edge.get("node_1", {}).get("name", "")
        if name1:
            span1 = find_text_span(original_text, name1)
            if span1:
                edge["node_1"]["text_span"] = {"start": span1[0], "end": span1[1]}

        # 节点 2 的索引 (精确匹配)
        name2 = edge.get("node_2", {}).get("name", "")
        if name2:
            span2 = find_text_span(original_text, name2)
            if span2:
                edge["node_2"]["text_span"] = {"start": span2[0], "end": span2[1]}

        # 关系/边的索引 (模糊匹配)
        relation = edge.get("relationship", "")
        if relation:
            # 针对关系使用模糊匹配
            span = find_fuzzy_match_span(original_text, relation, window=len(relation) + 5)
            if span:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["text_span"] = {"start": span[0], "end": span[1]}

    return edges

# --- LLM 提示词 (KG 提取) ---
system_prompt = (
    "You are an expert in extracting knowledge graphs from natural language text.\n"
    "Given an input text, extract all entities and relationships mentioned.\n"
    "The input text is a pair of contradictory sentences separated by [SEP]. Treat the entire input as one passage for extraction.\n"
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
    "IMPORTANT: Your response MUST STRICTLY be a JSON array. Do not include any text, markdown code fences (```json), or explanations outside of the array."
)


# --- 数据加载 (适配 JSON 数组结构) ---
def load_dense_text_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """尝试从 dense_text.json 文件中加载数据 (JSON 数组)。"""
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return None

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("🚨 错误：文件内容结构不符合预期的 JSON 数组格式。")
            return None

        print(f"✅ 成功加载 {len(data)} 条矛盾数据。")
        return data

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None


def process_single_doc(doc_entry: Dict[str, Any], llm: OpenAIClient, system_prompt: str) -> Optional[List[Dict[str, Any]]]:
    """处理单个文档条目，提取图谱并添加索引，加入重试机制和清洗逻辑。"""

    # 提取唯一ID和要处理的文本 (即合并后的矛盾句子)
    doc_id = doc_entry.get("source_id", "UNKNOWN_ID")
    full_text_input = doc_entry.get("text", "")

    if not full_text_input:
        return None

    print(f"\n--- 🛠️ 正在处理文档 ID: {doc_id} ---")

    for attempt in range(MAX_RETRIES):
        try:
            # 使用合并后的句子作为 LLM 输入
            response = llm.generate(user_message=full_text_input, system_message=system_prompt)

            # 🌟 增强的 JSON 清洗和提取逻辑
            json_text = response.strip()

            # 1. 移除 Markdown 围栏 (应对 LLM 自动添加的情况)
            # 使用非贪婪匹配移除可能存在的 '```json' 到 '```'
            match_cleaned = re.search(r"```(json)?\s*(.*?)\s*```", json_text, re.DOTALL)
            if match_cleaned:
                 json_text = match_cleaned.group(2).strip()

            # 如果没有围栏，进行简单的首尾清理
            else:
                 json_text = json_text.strip()

            # 2. 尝试清理 JSON 错误 (移除换行、制表符，并修正多余逗号)
            cleaned_json_text = json_text.replace('\n', '').replace('\t', '')
            # 移除对象或数组结束前的多余逗号 (例如: "key": "value",}] -> "key": "value"}] )
            cleaned_json_text = re.sub(r',\s*([\]}])', r'\1', cleaned_json_text)

            # 3. 提取完整的 JSON 数组 (确保是 [..] 结构)
            match = re.search(r"(\[.*\])", cleaned_json_text, re.DOTALL)

            if not match:
                # 抛出 ValueError 触发重试
                raise ValueError(f"LLM 响应中未找到有效 JSON 数组。")

            # 4. 安全加载
            edges = json.loads(match.group(1))

            # 5. 添加索引信息 (使用合并后的句子作为原始文本)
            edges_with_spans = add_text_spans_with_fuzzy_relation(edges, full_text_input)

            # 6. 添加文档ID等元数据
            for edge in edges_with_spans:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["source_id"] = doc_id
                # 添加矛盾类型作为边的属性，方便后续分析
                edge["edge_properties"]["contradiction_type"] = doc_entry.get("contradiction_type", "N/A")

            print(f"✅ 文档 {doc_id} 处理成功 (尝试 {attempt + 1})。提取 {len(edges_with_spans)} 条边。")
            return edges_with_spans

        except (json.JSONDecodeError, ValueError) as e:
            # 捕获 JSON 解析错误 或 提取失败
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️ 警告: 文档 {doc_id} 格式错误/提取失败 ({e})。正在重试... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(1) # 短暂等待
                continue
            else:
                print(f"🚨 错误: 文档 {doc_id} 格式错误，重试 {MAX_RETRIES} 次后失败 ({e})。跳过该文档。")
                return None
        except Exception as e:
            # 捕获其他未知错误（如 API 调用失败）
            print(f"🚨 错误: 处理文档 {doc_id} 时发生未知错误: {e}")
            return None
    return None

# --- 主执行逻辑 ---

def main():
    # 🎯 替换为加载 JSON 数组的函数
    dense_text_data = load_dense_text_data(INPUT_FILE)
    if not dense_text_data:
        return

    all_extracted_edges = []
    doc_count = len(dense_text_data)
    successful_docs = 0

    print(f"\n--- ⚙️ 开始知识图谱提取和索引添加 ---")

    for doc_entry in dense_text_data:
        edges = process_single_doc(doc_entry, llm, system_prompt)
        if edges:
            all_extracted_edges.extend(edges)
            successful_docs += 1

    failed_docs = doc_count - successful_docs

    print(f"\n--- 🎉 流程完成。 ---")
    print(f"   - 总输入条目数: {doc_count}")
    print(f"   - 成功提取图谱的条目数: {successful_docs}")
    print(f"   - 失败条目数 (重试后跳过): {failed_docs}")
    print(f"   - 总提取边数: {len(all_extracted_edges)}")

    # 打印格式化结果 (只打印前几个)
    print("\n[示例输出前 3 条边]:")
    print(json.dumps(all_extracted_edges[:3], indent=2, ensure_ascii=False))

    # Save to JSON file
    output_path = "dense_property_graph.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracted_edges, f, ensure_ascii=False, indent=2)

    print(f"\n知识图谱已保存至: {output_path}")

if __name__ == "__main__":
    main()