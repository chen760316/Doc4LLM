import os
import json
import time
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
import concurrent.futures
import threading

# --- 配置 ---
MAX_WORKERS = 10  # 并发线程数
MAX_RETRIES = 3  # LLM 最大重试次数
INPUT_FILE_PATH = "sampled_by_doc_type.json"
OUTPUT_FILE_PATH = "dense_text.json"  # 修改输出文件名以反映嵌套结构

# -------------------------
# 环境变量加载 (保持不变)
# -------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# -------------------------
# 初始化 LLM (保持不变)
# -------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 环境变量未设置。")

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.2,
    top_p=0.9,
    max_tokens=4000
)

# 用于保护打印输出的锁 (在多线程环境下保证输出不混乱)
print_lock = threading.Lock()

# -------------------------
# 系统提示词 (用于LLM提取)
# -------------------------
SYSTEM_PROMPT = """
You are an expert in analyzing documents to detect contradictions.

Your task is to read the provided single passage (which may be a concatenation of multiple sources) and identify **all pairs of sentences within that passage that contradict each other**.

Classify each contradiction into one of the following types: Factual, Content (Opinion/Emotion, Misinformation, Semantic/Lexical conflicts), Causal, Negation, Relation, Numeric, Temporal.

The two contradicting sentences must come from the passage itself.

### Output Format:
Return a JSON list of contradiction pairs:
[
  {
    "sentence_1": "...",
    "sentence_2": "...",
    "contradiction_type": "...",
    "reason": "..."
  },
  ...
]
If no contradictions are found, return the exact string: `No Contradiction Found.`
Be exhaustive — there may be multiple contradiction pairs.
"""


# -------------------------
# LLM 单任务调用及解析函数
# -------------------------
def call_llm_for_entry(unique_id: str, entry: Dict[str, Any], system_prompt: str) -> Optional[Dict[str, Any]]:
    """
    单个线程执行函数：调用 LLM，处理重试、JSON 解析和结果打包。
    返回一个字典，结构与输入 entry 相似，但包含提取结果。
    """
    full_text = entry.get("text", "")
    if not full_text:
        return {unique_id: {"status": "Skipped", "reason": "Empty text"}}

    user_prompt = f"Analyze the following passage and find all pairs of contradicting sentences within it:\n\nPassage:\n{full_text}"

    # --- 重试机制 ---
    max_retries = MAX_RETRIES
    retry_count = 0
    success = False
    contradictions = []

    # 打印初始调用信息
    with print_lock:
        print(f"[{threading.current_thread().name}] Calling LLM for ID: {unique_id}...")

    while retry_count < max_retries and not success:
        try:
            if retry_count > 0:
                with print_lock:
                    print(f"[{threading.current_thread().name}] 重试 ({retry_count}/{max_retries - 1}): ID {unique_id}")
                time.sleep(2 ** retry_count)

                # LLM 调用
            resp = llm.generate(user_message=user_prompt, system_message=system_prompt)
            result_str = resp.strip()

            if result_str == "No Contradiction Found.":
                success = True
                continue

            # --- 格式清理 ---
            result_str = result_str.replace("```json", "").replace("```", "").strip()

            # 尝试解析 JSON
            parsed_list = json.loads(result_str)
            if not isinstance(parsed_list, list):
                raise TypeError("JSON output is not a list.")

            contradictions = parsed_list
            success = True  # 解析成功，退出循环

        except json.JSONDecodeError as e:
            with print_lock:
                print(f"[{threading.current_thread().name}] JSON 解析失败 (尝试 {retry_count + 1}/{max_retries}): {e}")
                print(f"[{threading.current_thread().name}] LLM 原始输出 (用于调试): {result_str[:200]}...")
            retry_count += 1

        except Exception as e:
            with print_lock:
                print(f"[{threading.current_thread().name}] LLM 调用失败 (尝试 {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1

    # --- 结果打包 (确保一对一输出结构) ---
    result_entry = {
        "source_id": unique_id,
        "text": full_text,
        "doc_type": entry.get("doc_type", "N/A"),
        # 🎯 确保保留 scope 属性
        "scope": entry.get("scope", "N/A"),
        "contradiction_pairs": []
    }

    if not success:
        result_entry["status"] = "Failed (Max Retries)"
        return {unique_id: result_entry}

    if not contradictions:
        result_entry["status"] = "No Contradiction Found"
        return {unique_id: result_entry}

    result_entry["status"] = "Success"

    # 嵌套处理提取出的所有矛盾对
    for item in contradictions:
        sentence_1 = item.pop('sentence_1', '')
        sentence_2 = item.pop('sentence_2', '')

        # 🎯 核心修改点：将 sentence_1 和 sentence_2 合并到 'text' 字段
        pair_text = f"{sentence_1} [SEP] {sentence_2}"

        # 构建矛盾对的最终结构
        result_entry["contradiction_pairs"].append({
            "text": pair_text,
            "contradiction_type": item.get("contradiction_type", "N/A"),
            "reason": item.get("reason", "")
        })

    return {unique_id: result_entry}


# -------------------------
# LLM 动态提取函数 (多线程驱动)
# -------------------------
def extract_llm_contradictions_multithreaded(input_file: str, system_prompt: str) -> List[Dict[str, Any]]:
    """
    使用 ThreadPoolExecutor 并发处理所有输入案例。
    返回一个列表，其中每个元素对应输入文件中的一个 unique_id。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ 错误：加载文件 {input_file} 失败: {e}")
        return []

    pos_entries = data.get("pos", {})
    if not pos_entries:
        print("⚠️ 警告：输入文件中的 'pos' 字段为空或不存在。")
        return []

    all_llm_results = []

    futures_map = {}

    print(f"--- 🚀 启动多线程提取，共 {len(pos_entries)} 个样本，最大线程数 {MAX_WORKERS} ---")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for unique_id, entry in pos_entries.items():
            # 提交任务，将 entry 信息和 system_prompt 传递给工作函数
            future = executor.submit(call_llm_for_entry, unique_id, entry, system_prompt)
            futures_map[future] = unique_id

        # 收集结果
        for i, future in enumerate(concurrent.futures.as_completed(futures_map)):
            unique_id = futures_map[future]

            try:
                # future.result() 包含了线程返回的 {unique_id: result_entry} 字典
                result_dict = future.result()
                if result_dict:
                    all_llm_results.extend(list(result_dict.values()))
            except Exception as e:
                # 捕获线程执行中未处理的异常
                print(f"❌ 线程执行失败，ID {unique_id}：{e}")

            # 打印进度
            if (i + 1) % 50 == 0 or (i + 1) == len(pos_entries):
                with print_lock:
                    print(f"    [进度] 已处理 {i + 1}/{len(pos_entries)} 个样本...")

    return all_llm_results


# -------------------------
# 保存函数 (保持不变)
# -------------------------
def save_json_data(data: List[Dict[str, Any]], filename: str):
    """保存 JSON 格式的数据到指定文件"""
    # ⚠️ 由于输出结构变为了列表嵌套字典，这里使用列表保存
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"💾 Saved {len(data)} processed entries to {filename}")
    except Exception as e:
        print(f"⚠️ 错误：保存文件 {filename} 失败: {e}")


# -------------------------
# 主函数
# -------------------------
def main():
    print("\n--- 🧠 任务: 使用 LLM 动态提取自相矛盾句子 (多线程模式) ---")

    # 1. 使用 LLM 动态提取矛盾 (多线程)
    llm_extracted_results = extract_llm_contradictions_multithreaded(
        input_file=INPUT_FILE_PATH,
        system_prompt=SYSTEM_PROMPT
    )

    # 2. 统一保存结果
    if llm_extracted_results:
        save_json_data(llm_extracted_results, OUTPUT_FILE_PATH)
        print("\n--- ✅ 任务完成 ---")
        print(f"✨ 所有 LLM 处理后的案例 (共 {len(llm_extracted_results)} 个) 已保存到 {OUTPUT_FILE_PATH}。")
    else:
        print("\n--- ⚠️ 任务完成 ---")
        print("未提取到任何有效数据，未生成输出文件。")


if __name__ == "__main__":
    main()