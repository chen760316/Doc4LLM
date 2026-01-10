import os
import json
import re
import difflib
import time
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed  # 引入多线程模块

# --- 配置 ---
MAX_RETRIES = 3  # 最大重试次数
MAX_WORKERS = 12  # 线程池最大工作线程数 (可根据 API 限制和带宽调整)
INPUT_FILE = "dense_text.json"  # 输入文件路径
OUTPUT_PATH = "dense_property_graph.json"

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 初始化 LLM
from knowledge_graph_maker import OpenAIClient

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9
)


# --- 辅助函数 (保持不变) ---

def find_text_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    """找到 substring 在 text 中的第一个出现位置，返回 (start, end) 索引。"""
    match = re.search(re.escape(substring), text)
    if match:
        return (match.start(), match.end())
    return None


def find_fuzzy_match_span(text: str, pattern: str, window: int = 100) -> Optional[Tuple[int, int]]:
    """使用 difflib 进行模糊匹配，找到 pattern 在 text 中的最佳匹配范围。"""
    text_len = len(text)
    pattern_len = len(pattern)
    best_ratio = 0.9  # 设置一个合理的阈值
    best_span = None

    if pattern_len == 0 or pattern_len > text_len:
        return None

    for i in range(text_len - pattern_len + 1):
        segment = text[i:i + pattern_len]
        ratio = difflib.SequenceMatcher(None, segment, pattern).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_span = (i, i + pattern_len)

    return best_span


def add_text_spans_with_fuzzy_relation(edges: List[Dict[str, Any]], original_text: str) -> List[Dict[str, Any]]:
    """遍历提取的边，利用模糊匹配和精确匹配为节点和关系添加 text_span 索引。"""
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
            span = find_fuzzy_match_span(original_text, relation, window=len(relation) + 5)
            if span:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["text_span"] = {"start": span[0], "end": span[1]}

    return edges


# --- LLM 提示词 (保持不变) ---
system_prompt = (
    # ... (与原代码 SYSTEM_PROMPT 相同，保持简洁) ...
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


# --- 数据加载 (保持不变) ---
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


# --- 核心处理函数 (适配并发) ---
def process_single_doc_concurrent(
        doc_entry: Dict[str, Any],
        llm: OpenAIClient,
        system_prompt: str
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    处理单个文档条目，提取图谱并添加索引，加入重试机制和清洗逻辑。
    返回 (doc_id, 提取的边列表)。
    """

    doc_id = doc_entry.get("source_id", "UNKNOWN_ID")
    full_text_input = doc_entry.get("text", "")

    if not full_text_input:
        return doc_id, None

    # print(f"\n--- 🛠️ 正在处理文档 ID: {doc_id} ---") # 并发时减少打印次数

    for attempt in range(MAX_RETRIES):
        try:
            # 增加退避等待
            if attempt > 0:
                print(f"   重试 ({attempt + 1}/{MAX_RETRIES}): ID {doc_id}")
                time.sleep(2 ** attempt)

            response = llm.generate(user_message=full_text_input, system_message=system_prompt)

            # 🌟 增强的 JSON 清洗和提取逻辑
            json_text = response.strip()

            # 1. 移除 Markdown 围栏
            match_cleaned = re.search(r"```(json)?\s*(.*?)\s*```", json_text, re.DOTALL)
            if match_cleaned:
                json_text = match_cleaned.group(2).strip()
            else:
                json_text = json_text.strip()

            # 2. 尝试清理 JSON 错误
            cleaned_json_text = json_text.replace('\n', '').replace('\t', '')
            cleaned_json_text = re.sub(r',\s*([\]}])', r'\1', cleaned_json_text)

            # 3. 提取完整的 JSON 数组
            match = re.search(r"(\[.*\])", cleaned_json_text, re.DOTALL)

            if not match:
                raise ValueError(f"LLM 响应中未找到有效 JSON 数组。")

            # 4. 安全加载
            edges = json.loads(match.group(1))

            # 5. 添加索引信息
            edges_with_spans = add_text_spans_with_fuzzy_relation(edges, full_text_input)

            # 6. 添加文档ID等元数据
            contradiction_type = doc_entry.get("contradiction_type", "N/A")
            for edge in edges_with_spans:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["source_id"] = doc_id
                edge["edge_properties"]["contradiction_type"] = contradiction_type

            # print(f"✅ 文档 {doc_id} 处理成功 (尝试 {attempt + 1})。")
            return doc_id, edges_with_spans

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                continue
            else:
                print(f"🚨 错误: 文档 {doc_id} 格式错误，重试 {MAX_RETRIES} 次后失败 ({e})。跳过该文档。")
                return doc_id, None
        except Exception as e:
            print(f"🚨 错误: 处理文档 {doc_id} 时发生未知错误: {e}")
            return doc_id, None

    return doc_id, None  # 理论上循环结束后应该返回


# --- 主执行逻辑 (多线程实现) ---
def main():
    dense_text_data = load_dense_text_data(INPUT_FILE)
    if not dense_text_data:
        return

    all_extracted_edges = []
    doc_count = len(dense_text_data)
    successful_docs = 0
    failed_docs = 0
    processed_count = 0

    print(f"\n--- ⚙️ 启动线程池，文档总数: {doc_count}, 最大工作线程数: {MAX_WORKERS} ---")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_doc = {
            executor.submit(
                process_single_doc_concurrent, doc_entry, llm, system_prompt
            ): doc_entry.get("source_id", f"INDEX_{i}")  # 使用 source_id 或索引作为唯一标识符
            for i, doc_entry in enumerate(dense_text_data)
        }

        # 收集结果
        for future in as_completed(future_to_doc):
            doc_id = future_to_doc[future]
            processed_count += 1

            try:
                # future.result() 返回 (doc_id, edges)
                _, edges = future.result()
                if edges:
                    all_extracted_edges.extend(edges)
                    successful_docs += 1
                    print(f"✅ ID {doc_id} 提取成功 ({len(edges)} 条边)。进度: {processed_count}/{doc_count}")
                else:
                    failed_docs += 1
                    print(f"➖ ID {doc_id} 提取失败或无边。进度: {processed_count}/{doc_count}")

            except Exception as exc:
                print(f'🚨 ID {doc_id} 任务生成了一个未捕获的异常: {exc}')
                failed_docs += 1

    print(f"\n--- 🎉 流程完成。 ---")
    print(f"   - 总输入条目数: {doc_count}")
    print(f"   - 成功提取图谱的条目数: {successful_docs}")
    print(f"   - 失败条目数 (重试后跳过或异常): {failed_docs}")
    print(f"   - 总提取边数: {len(all_extracted_edges)}")

    # 打印格式化结果 (只打印前几个)
    print("\n[示例输出前 3 条边]:")
    print(json.dumps(all_extracted_edges[:3], indent=2, ensure_ascii=False))

    # Save to JSON file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_extracted_edges, f, ensure_ascii=False, indent=2)

    print(f"\n知识图谱已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()