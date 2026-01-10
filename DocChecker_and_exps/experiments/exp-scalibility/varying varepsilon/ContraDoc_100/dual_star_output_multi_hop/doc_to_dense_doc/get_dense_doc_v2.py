"""
========================================
📊 运行报告:
⏱️ 总耗时: 60.51 秒
📄 处理条目数: 194
🎯 提取总矛盾对: 273
⚡ 平均处理速度: 0.31 秒/条目
💾 结果保存至: dense_text.json
========================================
"""
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
from typing import Dict, Any, List

# -------------------------
# 配置与初始化
# -------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

MAX_WORKERS = 10  # 线程池大小，根据你的 API 限制调整
INPUT_FILE_PATH = "../../sampled_data.json"
OUTPUT_FILE_PATH = "dense_text.json"

llm = OpenAIClient(
    model="gpt-5.2",
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
# 单个条目处理函数 (供线程调用)
# -------------------------
def process_single_entry(unique_id: str, entry: Dict[str, Any], system_prompt: str) -> List[Dict[str, Any]]:
    full_text = entry.get("text", "")
    if not full_text:
        return []

    user_prompt = f"Analyze the following passage and find all pairs of contradicting sentences within it:\n\nPassage:\n{full_text}"

    max_retries = 3
    for retry_count in range(max_retries):
        try:
            if retry_count > 0:
                time.sleep(1)  # 简单重试等待

            resp = llm.generate(user_message=user_prompt, system_message=system_prompt)
            result_str = resp.strip()

            if result_str == "No Contradiction Found.":
                return []

            # 格式清理
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0]
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0]

            result_str = result_str.strip()
            contradictions = json.loads(result_str)

            # 数据转换与元数据注入
            processed_items = []
            for item in contradictions:
                s1 = item.pop('sentence_1', '')
                s2 = item.pop('sentence_2', '')
                processed_items.append({
                    "text": f"{s1} [SEP] {s2}",
                    "contradiction_type": item.get("contradiction_type", ""),
                    "reason": item.get("reason", ""),
                    "source_id": unique_id,
                    "extraction_method": "LLM_Self_Contradiction",
                    "doc_type": entry.get("doc_type", "N/A")
                })

            print(f"✅ ID {unique_id}: 提取到 {len(processed_items)} 处矛盾")
            return processed_items

        except Exception as e:
            if retry_count == max_retries - 1:
                print(f"❌ ID {unique_id} 处理失败 (尝试 {retry_count + 1}次): {e}")

    return []


# -------------------------
# 主逻辑
# -------------------------
def main():
    print(f"\n--- 🧠 任务: 多线程动态提取矛盾 (并发数: {MAX_WORKERS}) ---")

    # 加载数据
    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ 无法加载输入文件: {e}")
        return

    pos_entries = data.get("pos", {})
    all_llm_results = []

    # 计时开始
    start_time = time.time()

    # 使用线程池并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_id = {
            executor.submit(process_single_entry, uid, entry, SYSTEM_PROMPT): uid
            for uid, entry in pos_entries.items()
        }

        # 收集结果
        for future in as_completed(future_to_id):
            result = future.result()
            if result:
                all_llm_results.extend(result)

    # 计时结束
    end_time = time.time()
    total_duration = end_time - start_time

    # 保存结果
    if all_llm_results:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_llm_results, f, ensure_ascii=False, indent=4)

    # 打印运行报告
    print("\n" + "=" * 40)
    print(f"📊 运行报告:")
    print(f"⏱️ 总耗时: {total_duration:.2f} 秒")
    print(f"📄 处理条目数: {len(pos_entries)}")
    print(f"🎯 提取总矛盾对: {len(all_llm_results)}")
    if len(pos_entries) > 0:
        print(f"⚡ 平均处理速度: {total_duration / len(pos_entries):.2f} 秒/条目")
    print(f"💾 结果保存至: {OUTPUT_FILE_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    main()