"""
========================================
📊 运行报告:
⏱️ 总耗时: 276.07 秒
📄 处理条目数: 450
🎯 提取总矛盾对: 2037
⚡ 平均处理速度: 0.61 秒/条目
💾 结果保存至: dense_text.json
========================================
"""
import os
import json
import time
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
from typing import Dict, Any, List, Union

# -------------------------
# 配置与初始化
# -------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

MAX_WORKERS = 10
INPUT_FILE_PATH = "../../sampled_data.json"
OUTPUT_FILE_PATH = "dense_text.json"

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.2,
    top_p=0.9,
    max_tokens=4000
)

SYSTEM_PROMPT = """
You are an expert in analyzing documents to detect contradictions.
Identify all pairs of sentences within that passage that contradict each other.
Return a JSON list:
[
  {"sentence_1": "...", "sentence_2": "...", "contradiction_type": "...", "reason": "..."}
]
If none, return: No Contradiction Found.
"""

# -------------------------
# 单个条目处理函数
# -------------------------
def process_single_entry(unique_id: str, entry: Dict[str, Any], system_prompt: str) -> List[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return []

    full_text = entry.get("text", "")
    if not full_text:
        return []

    user_prompt = f"Analyze the following passage and find all pairs of contradicting sentences within it:\n\nPassage:\n{full_text}"

    max_retries = 3
    for retry_count in range(max_retries):
        try:
            if retry_count > 0:
                time.sleep(1)

            resp = llm.generate(user_message=user_prompt, system_message=system_prompt)
            result_str = resp.strip()

            if "No Contradiction Found" in result_str:
                return []

            # 增强型 JSON 清理：处理 Markdown 代码块
            if "```" in result_str:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result_str)
                if match:
                    result_str = match.group(1).strip()

            contradictions = json.loads(result_str)
            if not isinstance(contradictions, list):
                return []

            processed_items = []
            for item in contradictions:
                # 使用 pop 确保字段被提取，同时设置默认值防崩
                s1 = item.get('sentence_1', '')
                s2 = item.get('sentence_2', '')
                processed_items.append({
                    "text": f"{s1} [SEP] {s2}",
                    "contradiction_type": item.get("contradiction_type", ""),
                    "reason": item.get("reason", ""),
                    "source_id": unique_id,
                    "extraction_method": "LLM_Self_Contradiction",
                    "doc_type": entry.get("doc_type", "N/A"),
                    "label_text": "contradiction" # 注入标签方便后续处理
                })

            return processed_items

        except Exception as e:
            if retry_count == max_retries - 1:
                print(f"❌ ID {unique_id} 处理失败: {e}")

    return []

# -------------------------
# 主逻辑
# -------------------------
def main():
    print(f"\n--- 🧠 任务: 多线程动态提取矛盾 (并发数: {MAX_WORKERS}) ---")

    # 1. 加载并检查数据格式
    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ 无法加载输入文件: {e}")
        return

    # 核心修复：动态适配 List 或 Dict 格式
    task_items = []
    if isinstance(data, list):
        print(f"💡 检测到输入为列表格式 (List)，共 {len(data)} 条。")
        task_items = [(f"idx_{i}", entry) for i, entry in enumerate(data)]
    elif isinstance(data, dict):
        print("💡 检测到输入为字典格式 (Dict)。")
        pos_entries = data.get("pos", {})
        if isinstance(pos_entries, dict):
            task_items = list(pos_entries.items())
        elif isinstance(pos_entries, list):
            task_items = [(f"idx_{i}", entry) for i, entry in enumerate(pos_entries)]
    else:
        print(f"🚨 错误：不支持的数据格式 {type(data)}")
        return

    if not task_items:
        print("🚨 没有找到有效的数据条目，请检查 JSON 结构。")
        return

    all_llm_results = []
    start_time = time.time()

    # 2. 并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(process_single_entry, uid, entry, SYSTEM_PROMPT): uid
            for uid, entry in task_items
        }

        for future in as_completed(future_to_id):
            uid = future_to_id[future]
            try:
                result = future.result()
                if result:
                    all_llm_results.extend(result)
                    print(f"✅ ID {uid}: 提取到 {len(result)} 处矛盾")
            except Exception as e:
                print(f"🚨 线程处理异常 (ID {uid}): {e}")

    # 3. 汇总与保存
    total_duration = time.time() - start_time

    if all_llm_results:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_llm_results, f, ensure_ascii=False, indent=4)

    # 打印运行报告
    print("\n" + "=" * 40)
    print(f"📊 运行报告:")
    print(f"⏱️ 总耗时: {total_duration:.2f} 秒")
    print(f"📄 处理条目数: {len(task_items)}")
    print(f"🎯 提取总矛盾对: {len(all_llm_results)}")
    if task_items:
        print(f"⚡ 平均处理速度: {total_duration / len(task_items):.2f} 秒/条目")
    print(f"💾 结果保存至: {OUTPUT_FILE_PATH}")
    print("=" * 40)

if __name__ == "__main__":
    main()