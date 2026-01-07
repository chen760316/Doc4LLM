"""
========================================
🎉 任务提前完成 (熔断返回)
----------------------------------------
🚀 总耗时: 80.73 秒
处理文档数: 174 / 258
最终唯一节点数: 400
提取总边数: 364
平均每产生 1 个节点耗时: 0.2018 秒
========================================
"""
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
import difflib
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed # 引入多线程模块
import time
import threading # 引入线程锁

# --- 配置 ---
MAX_RETRIES = 3 # 最大重试次数
MAX_WORKERS = 10 # 线程池最大工作线程数
NODE_LIMIT = 400 # 🎯 节点限制阈值

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 升级到 gpt-4o-mini
llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9
)

# --- 全局状态管理 ---
data_lock = threading.Lock()      # 保护全局列表和集合
unique_nodes = set()             # 全局唯一节点名集合
all_extracted_edges = []         # 全局提取结果
stop_event = threading.Event()   # 停止信号

# --- 配置 ---
INPUT_FILE = "dense_text.json" # 假设输入文件名为 sampled_data.json

# --- 数据加载函数 (更新为新格式) ---
def load_new_format_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """尝试从新的 JSON 文件中加载数据（列表格式）。"""
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return None

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("🚨 错误：文件内容结构不符合预期的 JSON 列表格式。")
            return None

        print(f"✅ 成功加载 {len(data)} 条数据。")
        return data

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None

# --- 索引查找函数 (保持不变) ---
def find_text_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    """找到 substring 在 text 中的第一个出现位置，返回 (start, end) 索引。"""
    match = re.search(re.escape(substring), text)
    if match:
        return (match.start(), match.end())
    else:
        return None

def find_fuzzy_match_span(text: str, pattern: str, window: int = 100) -> Optional[Tuple[int, int]]:
    """模糊匹配关系文本的索引。"""
    # ... (函数体保持不变) ...
    text_len = len(text)
    pattern_len = len(pattern)
    best_ratio = 0
    best_span = (0, 0)

    if pattern_len == 0 or pattern_len > text_len:
        return None

    step = max(10, pattern_len // 2)
    search_window = max(pattern_len, window)

    for start in range(0, text_len, step):
        end = min(start + search_window, text_len)
        segment = text[start:end]
        ratio = difflib.SequenceMatcher(None, segment, pattern).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_start = segment.find(pattern)
            if best_match_start != -1:
                 best_span = (start + best_match_start, start + best_match_start + pattern_len)
            else:
                 best_span = (start, end)
        if best_ratio > 0.95:
            break

    return best_span if best_ratio > 0 else None


def add_text_spans_with_fuzzy_relation(edges: List[Dict[str, Any]], original_text: str) -> List[Dict[str, Any]]:
    """为节点和关系添加在原始文本中的索引。"""
    for edge in edges:
        # 节点 1 索引
        name1 = edge["node_1"].get("name", "")
        span1 = find_text_span(original_text, name1)
        if span1:
            edge["node_1"]["text_span"] = {"start": span1[0], "end": span1[1]}

        # 节点 2 索引
        name2 = edge["node_2"].get("name", "")
        span2 = find_text_span(original_text, name2)
        if span2:
            edge["node_2"]["text_span"] = {"start": span2[0], "end": span2[1]}

        # 关系索引 (模糊匹配)
        relation = edge.get("relationship", "")
        if relation:
            span = find_fuzzy_match_span(original_text, relation)
            if span:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["text_span"] = {"start": span[0], "end": span[1]}

    return edges


system_prompt = (
    "You are an expert in extracting knowledge graphs from natural language text.\n"
    "Given an input text, extract all entities and relationships mentioned.\n"
    # ... (Prompt其余部分保持不变) ...
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


# 🛡️ 增强的线程安全处理函数
def process_single_doc_task(
        doc_index: int,
        doc_data: Dict[str, Any],
        llm: OpenAIClient,
        system_prompt: str
) -> Optional[List[Dict[str, Any]]]:
    """处理单个文档，并实时更新全局节点计数。"""

    # 1. 检查是否已经达到阈值，如果达标则直接退出
    if stop_event.is_set():
        return None

    # 兼容原逻辑：合并文本
    conflicting_texts = doc_data.get("text", [])
    if isinstance(conflicting_texts, list):
        original_text = "\n".join(conflicting_texts).strip()
    else:
        original_text = str(conflicting_texts).strip()

    if not original_text:
        return None

    for attempt in range(MAX_RETRIES):
        if stop_event.is_set(): return None  # 重试前再次检查信号

        try:
            response = llm.generate(user_message=original_text, system_message=system_prompt)
            json_text = response.strip()

            # --- JSON 清理逻辑 (保持不变) ---
            if json_text.startswith("```"):
                json_text = re.sub(r"```(json)?\s*", "", json_text, flags=re.DOTALL).strip()
                if json_text.endswith("```"): json_text = json_text[:-3].strip()
            cleaned_json_text = json_text.replace('\n', '').replace('\t', '')
            cleaned_json_text = re.sub(r',\s*([\]}])', r'\1', cleaned_json_text)
            match = re.search(r"(\[.*\])", cleaned_json_text, re.DOTALL)
            if not match: raise ValueError("Invalid JSON format")

            edges = json.loads(match.group(1))
            edges = add_text_spans_with_fuzzy_relation(edges, original_text)

            # --- 🎯 关键：线程安全更新与熔断检查 ---
            local_new_nodes = set()
            for edge in edges:
                # 注入元数据
                if "edge_properties" not in edge: edge["edge_properties"] = {}
                edge["edge_properties"]["doc_index"] = doc_index
                # 收集本批次的节点名用于计数
                local_new_nodes.add(edge["node_1"].get("name", ""))
                local_new_nodes.add(edge["node_2"].get("name", ""))

            with data_lock:
                if stop_event.is_set(): return None  # 双重检查防止越界

                all_extracted_edges.extend(edges)
                unique_nodes.update(local_new_nodes)

                current_count = len(unique_nodes)
                print(f"📈 [线程 {threading.get_ident()}] 文档 {doc_index} 成功 | 当前总节点数: {current_count}")

                if current_count >= NODE_LIMIT:
                    print(f"🎯 节点数已达标 ({current_count} >= {NODE_LIMIT})，正在触发熔断...")
                    stop_event.set()

            return edges

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"🚨 文档 {doc_index} 最终失败: {e}")
            continue

    return None


# --- 主执行逻辑 ---

def main():
    new_format_data = load_new_format_data(INPUT_FILE)
    if not new_format_data: return

    start_time = time.perf_counter()
    doc_count = len(new_format_data)
    processed_docs = 0

    print(f"\n--- ⚙️ 启动多线程任务 | 目标节点数: {NODE_LIMIT} | 并发数: {MAX_WORKERS} ---")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务，但由于 stop_event 的存在，多余任务会快速跳过
        future_to_doc = {
            executor.submit(process_single_doc_task, i, data, llm, system_prompt): i
            for i, data in enumerate(new_format_data)
        }

        # 动态收集结果，一旦信号触发即停止处理后续 Future
        for future in as_completed(future_to_doc):
            processed_docs += 1
            if stop_event.is_set():
                # 尝试取消还没开始的任务
                executor.shutdown(wait=False, cancel_futures=True)
                break

    duration = time.perf_counter() - start_time

    print(f"\n" + "=" * 40)
    print(f"🎉 任务提前完成 (熔断返回)")
    print("-" * 40)
    print(f"🚀 总耗时: {duration:.2f} 秒")
    print(f"处理文档数: {processed_docs} / {doc_count}")
    print(f"最终唯一节点数: {len(unique_nodes)}")
    print(f"提取总边数: {len(all_extracted_edges)}")

    if len(unique_nodes) > 0:
        print(f"平均每产生 1 个节点耗时: {duration / len(unique_nodes):.4f} 秒")
    print("=" * 40)

    # 保存结果
    output_path = "dense_property_graph.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracted_edges, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 结果已保存至: {output_path}")


if __name__ == "__main__":
    main()