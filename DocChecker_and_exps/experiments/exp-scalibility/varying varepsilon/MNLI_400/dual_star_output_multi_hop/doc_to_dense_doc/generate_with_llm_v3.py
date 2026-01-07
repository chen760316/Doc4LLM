"""
========================================
📊 任务统计报告
----------------------------------------
🚀 总耗时: 49.08 秒
处理文档数: 106 / 2037
最终唯一节点数: 400
提取总边数: 351
平均每产生 1 个节点耗时: 0.1227 秒
========================================
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
import difflib
from typing import Dict, Any, Optional, List, Tuple
import time

# --- 配置 ---
MAX_RETRIES = 3
MAX_WORKERS = 10
NODE_LIMIT = 400       # 🎯 节点限制阈值
INPUT_FILE = "dense_text.json" # 你的输入文件名

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(model="gpt-4o-mini", temperature=0.1, top_p=0.9)

# --- 全局状态管理 ---
data_lock = threading.Lock()
unique_nodes = set()
all_extracted_edges = []
stop_event = threading.Event()

# --- 辅助函数：索引查找与模糊匹配 ---
def find_text_span(text: str, substring: str):
    if not substring: return None
    match = re.search(re.escape(substring), text)
    return (match.start(), match.end()) if match else None

def find_fuzzy_match_span(text: str, pattern: str, window: int = 100):
    text_len, pattern_len = len(text), len(pattern)
    if pattern_len == 0 or pattern_len > text_len: return None
    best_ratio, best_span = 0, (0, 0)
    step, search_window = max(10, pattern_len // 2), max(pattern_len, window)
    for start in range(0, text_len, step):
        end = min(start + search_window, text_len)
        segment = text[start:end]
        ratio = difflib.SequenceMatcher(None, segment, pattern).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            idx = segment.find(pattern)
            best_span = (start + idx, start + idx + pattern_len) if idx != -1 else (start, end)
        if best_ratio > 0.95: break
    return best_span if best_ratio > 0 else None

def add_text_spans_with_fuzzy_relation(edges, original_text):
    for edge in edges:
        if not isinstance(edge, dict): continue
        for node_key in ["node_1", "node_2"]:
            if node_key in edge:
                name = edge[node_key].get("name", "")
                span = find_text_span(original_text, name)
                if span: edge[node_key]["text_span"] = {"start": span[0], "end": span[1]}
        relation = edge.get("relationship", "")
        if relation:
            span = find_fuzzy_match_span(original_text, relation)
            if span:
                if "edge_properties" not in edge: edge["edge_properties"] = {}
                edge["edge_properties"]["text_span"] = {"start": span[0], "end": span[1]}
    return edges

# --- 核心处理逻辑 ---
def process_single_doc_task(doc_index: int, doc_data: Dict[str, Any], system_prompt: str):
    """处理单个文档，并实时更新全局节点计数。"""
    if stop_event.is_set(): return None

    # 🎯 处理 [SEP] 分隔符，将其转为换行符以增强 LLM 理解力
    raw_text = doc_data.get("text", "")
    if not raw_text: return None
    original_text = raw_text.replace(" [SEP] ", "\n").replace("[SEP]", "\n")

    for attempt in range(MAX_RETRIES):
        if stop_event.is_set(): return None

        try:
            response = llm.generate(user_message=original_text, system_message=system_prompt)
            json_text = response.strip()

            # JSON 清理
            if json_text.startswith("```"):
                json_text = re.sub(r"```(json)?\s*", "", json_text, flags=re.DOTALL).strip()
                if json_text.endswith("```"): json_text = json_text[:-3].strip()

            cleaned_json_text = json_text.replace('\n', '').replace('\t', '')
            cleaned_json_text = re.sub(r',\s*([\]}])', r'\1', cleaned_json_text)
            match = re.search(r"(\[.*\])", cleaned_json_text, re.DOTALL)
            if not match: raise ValueError("No valid JSON array")

            edges = json.loads(match.group(1))
            edges = add_text_spans_with_fuzzy_relation(edges, original_text)

            # --- 🎯 线程安全更新与熔断检查 ---
            local_new_nodes = set()
            for edge in edges:
                if "edge_properties" not in edge: edge["edge_properties"] = {}
                edge["edge_properties"]["doc_index"] = doc_index
                edge["edge_properties"]["label_text"] = doc_data.get("label_text", "unknown")

                local_new_nodes.add(edge.get("node_1", {}).get("name", ""))
                local_new_nodes.add(edge.get("node_2", {}).get("name", ""))

            with data_lock:
                if stop_event.is_set(): return None

                all_extracted_edges.extend(edges)
                unique_nodes.update({n for n in local_new_nodes if n}) # 过滤空名

                current_count = len(unique_nodes)
                print(f"📈 [Doc {doc_index}] 成功 | 当前唯一节点总数: {current_count}")

                if current_count >= NODE_LIMIT:
                    print(f"🎯 熔断：节点数已达标 ({current_count} >= {NODE_LIMIT})")
                    stop_event.set()

            return edges

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"🚨 文档 {doc_index} 失败: {e}")
    return None

def main():
    system_prompt = (
        "Extract a knowledge graph as a JSON array. Each element MUST contain 'node_1', 'node_2', and 'relationship'. "
        "For each entity, include 'name' and 'label'. Provide natural language relationships."
    )

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 文件不存在: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"--- 🚀 启动多线程任务 | 目标: {NODE_LIMIT} 节点 | 并发: {MAX_WORKERS} ---")

    start_time = time.perf_counter()
    processed_docs = 0



    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_doc = {
            executor.submit(process_single_doc_task, i, item, system_prompt): i
            for i, item in enumerate(data_list)
        }

        # 动态收集结果，支持提前退出
        for future in as_completed(future_to_doc):
            processed_docs += 1
            if stop_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                break

    duration = time.perf_counter() - start_time

    print(f"\n" + "=" * 40)
    print(f"📊 任务统计报告")
    print("-" * 40)
    print(f"🚀 总耗时: {duration:.2f} 秒")
    print(f"处理文档数: {processed_docs} / {len(data_list)}")
    print(f"最终唯一节点数: {len(unique_nodes)}")
    print(f"提取总边数: {len(all_extracted_edges)}")
    if unique_nodes:
        print(f"平均每产生 1 个节点耗时: {duration / len(unique_nodes):.4f} 秒")
    print("=" * 40)

    # 保存结果
    output_path = "dense_property_graph.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracted_edges, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 结果已保存至: {output_path}")

if __name__ == "__main__":
    main()