"""
==================================================
📊 提取报告
⏱️ 时间: 34.71s | ✅ 文档: 9 | 🧬 节点: 102 | 🔗 边: 119
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
NODE_LIMIT = 100
INPUT_FILE = "sampled_data.json"

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(model="gpt-5.2", temperature=0.1, top_p=0.9)

data_lock = threading.Lock()
all_extracted_edges = []
unique_nodes = set()
stop_event = threading.Event()


# --- 辅助函数 ---
def load_contra_doc_data(file_path: str):
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None


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
        ratio = difflib.SequenceMatcher(None, text[start:end], pattern).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            idx = text[start:end].find(pattern)
            best_span = (start + idx, start + idx + pattern_len) if idx != -1 else (start, end)
        if best_ratio > 0.95: break
    return best_span if best_ratio > 0 else None


def add_text_spans_with_fuzzy_relation(edges, original_text):
    valid_edges = []
    for edge in edges:
        # 🛡️ 增强校验：确保 edge 是字典且包含 node_1 和 node_2
        if not isinstance(edge, dict) or "node_1" not in edge or "node_2" not in edge:
            continue

        for node_key in ["node_1", "node_2"]:
            node_data = edge[node_key]
            if isinstance(node_data, dict):
                name = node_data.get("name", "")
                span = find_text_span(original_text, name)
                if span: node_data["text_span"] = {"start": span[0], "end": span[1]}

        relation = edge.get("relationship", "")
        if relation:
            span = find_fuzzy_match_span(original_text, relation)
            if span:
                if "edge_properties" not in edge: edge["edge_properties"] = {}
                edge["edge_properties"]["text_span"] = {"start": span[0], "end": span[1]}
        valid_edges.append(edge)
    return valid_edges


# --- 核心处理逻辑 ---

def process_single_doc_task(doc_id: str, doc_data: Dict[str, Any], system_prompt: str):
    if stop_event.is_set(): return None

    original_text = doc_data.get("text", "")
    if not original_text: return None

    for attempt in range(MAX_RETRIES):
        try:
            response = llm.generate(user_message=original_text, system_message=system_prompt)
            json_text = response.strip()

            # JSON 清洗逻辑
            if json_text.startswith("```"):
                json_text = re.sub(r"```(json)?\s*", "", json_text, flags=re.DOTALL).strip()
                if json_text.endswith("```"): json_text = json_text[:-3].strip()

            cleaned_json_text = json_text.replace('\n', '').replace('\t', '')
            cleaned_json_text = re.sub(r',\s*([\]}])', r'\1', cleaned_json_text)
            match = re.search(r"(\[.*\])", cleaned_json_text, re.DOTALL)

            if not match: raise ValueError("No valid JSON array found")

            raw_edges = json.loads(match.group(1))

            # 🛡️ 在这里进行结构校验，防止 KeyError: 'node_1'
            edges = add_text_spans_with_fuzzy_relation(raw_edges, original_text)

            for edge in edges:
                if "edge_properties" not in edge: edge["edge_properties"] = {}
                edge["edge_properties"]["doc_id"] = doc_id

            with data_lock:
                if stop_event.is_set(): return edges

                all_extracted_edges.extend(edges)
                for edge in edges:
                    # 使用 .get() 并检查 None 值，防止程序崩溃
                    n1 = edge.get("node_1", {}).get("name", "").strip()
                    n2 = edge.get("node_2", {}).get("name", "").strip()
                    if n1: unique_nodes.add(n1)
                    if n2: unique_nodes.add(n2)

                count = len(unique_nodes)
                print(f"📈 进度: {doc_id} | 节点数: {count}")

                if count >= NODE_LIMIT:
                    print(f"🎯 达标熔断: {count} 节点。")
                    stop_event.set()

            return edges

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                # 打印详细错误信息有助于调试
                print(f"🚨 文档 {doc_id} 失败: {type(e).__name__} - {e}")
    return None


def main():
    system_prompt = (
        "Extract a knowledge graph as a JSON array. Each element MUST contain 'node_1', 'node_2', and 'relationship'. "
        "Example: [{\"node_1\": {\"name\": \"A\", \"label\": \"Person\"}, \"node_2\": {\"name\": \"B\"}, \"relationship\": \"works for\"}]"
    )

    data_dict = load_contra_doc_data(INPUT_FILE)
    if not data_dict: return

    task_list = []
    for doc_type in ['pos', 'neg']:
        if doc_type in data_dict:
            for doc_id, doc_data in data_dict[doc_type].items():
                task_list.append((doc_id, doc_data))

    print(f"🚀 启动多线程: {MAX_WORKERS} 线程 | 目标: {NODE_LIMIT} 节点")

    start_time = time.time()
    successful_docs = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_doc_task, tid, tdata, system_prompt): tid for tid, tdata in task_list}

        for future in as_completed(futures):
            if future.result(): successful_docs += 1
            if stop_event.is_set():
                # 尝试取消还未开始的任务
                for f in futures: f.cancel()
                break

    duration = time.time() - start_time
    print(f"\n" + "=" * 50)
    print(f"📊 提取报告\n⏱️ 时间: {duration:.2f}s | ✅ 文档: {successful_docs} | 🧬 节点: {len(unique_nodes)} | 🔗 边: {len(all_extracted_edges)}")
    print("=" * 50)

    with open("output_knowledge_graph_v3.json", "w", encoding="utf-8") as f:
        json.dump(all_extracted_edges, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()