"""
==============================
      ⏰ 运行时间统计
==============================
总执行耗时: 16.59 秒
平均处理速度: 1.66 秒/文档
任务结束时间: 2025-12-23 17:57:40
==============================
"""
import os
import json
import time
import re
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# 配置
# -------------------------
MAX_WORKERS = 10  # 线程池最大工作线程数
MAX_RETRIES = 3  # LLM调用最大重试次数

# -------------------------
# 环境变量加载 (保持不变)
# -------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# -------------------------
# 初始化 LLM (保持不变)
# -------------------------
llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.2,
    top_p=0.9,
    max_tokens=4000
)

# -------------------------
# 系统提示词 (保持不变)
# -------------------------
SYSTEM_PROMPT = """
You are an expert in analyzing documents to detect contradictions.

Your task is to read the provided single passage (which may be a concatenation of multiple sources) and identify **all pairs of sentences within that passage that contradict each other**.

Classify each contradiction into one of the following types: Factual, Content (Opinion/Emotion, Misinformation, Semantic/Lexical conflicts), Causal, Negation, Relation, Numeric, Temporal.

The two contradicting sentences must come from the passage itself.

### Output Format:
Return a JSON list:
[
  {
    "sentence_1": "...",
    "sentence_2": "...",
    "contradiction_type": "...",
    "reason": "..."
  },
  ...
]
If no contradictions are found, return: `No Contradiction Found.`
Be exhaustive — there may be multiple contradiction pairs.
"""


# -------------------------
# 单个文档处理函数 (适配新数据结构)
# -------------------------
def process_single_entry_new_format(
        doc_index: int, # 使用索引作为唯一标识符
        entry: Dict[str, Any],
        llm: OpenAIClient,
        system_prompt: str
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    处理单个文档条目，从 'conflicting_texts' 提取文本，调用 LLM 提取矛盾，并执行重试和格式清理。
    返回 (doc_index, 提取的矛盾列表)。
    """
    # 🎯 核心修改 1: 从 conflicting_texts 提取文本
    conflicting_texts = entry.get("text", [])
    full_text = "\n".join(conflicting_texts).strip()

    if not full_text:
        print(f"⚠️ 警告: 文档索引 {doc_index} 的 'conflicting_texts' 为空。跳过。")
        return doc_index, []

    user_prompt = f"Analyze the following passage and find all pairs of contradicting sentences within it:\n\nPassage:\n{full_text}"
    print(f"Calling LLM for Index: {doc_index}...")

    # --- 重试机制 (保持不变) ---
    retry_count = 0
    success = False
    result_str = ""

    while retry_count < MAX_RETRIES and not success:
        try:
            if retry_count > 0:
                print(f"   重试 ({retry_count}/{MAX_RETRIES - 1}): Index {doc_index}")
                time.sleep(2 ** retry_count)

            resp = llm.generate(user_message=user_prompt, system_message=system_prompt)
            result_str = resp.strip()

            if result_str == "No Contradiction Found.":
                success = True
                continue

            # --- 🎯 格式清理核心 (保持不变) ---
            if result_str.startswith("```json"):
                result_str = result_str[7:]
            if result_str.endswith("```"):
                result_str = result_str[:-3]
            result_str = result_str.strip()
            # -------------------------

            # 尝试解析 JSON
            contradictions = json.loads(result_str)
            success = True

        except json.JSONDecodeError:
            retry_count += 1
        except Exception:
            retry_count += 1

    # --- 结果处理 ---
    if not success:
        print(f"❌ 达到最大重试次数，Index {doc_index} 处理失败。")
        return doc_index, []

    if result_str == "No Contradiction Found.":
        return doc_index, []

    # 确保 contradictions 变量在成功时被设置
    if success and 'contradictions' in locals() and isinstance(contradictions, list):
        final_results = []
        for item in contradictions:
            # 🎯 核心修改 2: 更新元数据
            item["source_index"] = doc_index # 使用文档索引
            item["source_main_entity"] = entry.get("main_entity", "N/A")
            item["source_domain"] = entry.get("source_domain", "N/A")
            item["extraction_method"] = "LLM_Self_Contradiction"

            # 🎯 核心修改 3: 合并 sentence_1 和 sentence_2 到 'text' 字段
            sentence_1 = item.pop('sentence_1', '')
            sentence_2 = item.pop('sentence_2', '')
            item["text"] = f"{sentence_1} [SEP] {sentence_2}"

            final_results.append(item)

        return doc_index, final_results

    return doc_index, []


# -------------------------
# LLM 动态提取函数 (适配新数据结构)
# -------------------------
def extract_llm_contradictions_concurrent_new_format(input_file: str, llm: OpenAIClient, system_prompt: str) -> List[Dict[str, Any]]:
    """
    加载新 JSON 列表格式，并使用 ThreadPoolExecutor 并发处理。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ 错误：加载文件 {input_file} 失败: {e}")
        return []

    # 🎯 核心修改 4: 检查数据是否为列表
    if not isinstance(data, list):
        print(f"⚠️ 错误：文件 {input_file} 内容不是预期的 JSON 列表格式。")
        return []

    all_extracted_contradictions = []

    doc_list = data
    doc_count = len(doc_list)
    processed_count = 0
    successful_count = 0

    print(f"\n--- ⚙️ 启动线程池，文档总数: {doc_count}, 最大工作线程数: {MAX_WORKERS} ---")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 🎯 核心修改 5: 迭代列表和索引
        future_to_index = {
            executor.submit(
                process_single_entry_new_format, i, entry, llm, system_prompt
            ): i
            for i, entry in enumerate(doc_list)
        }

        # 收集结果
        for future in as_completed(future_to_index):
            doc_index = future_to_index[future]
            processed_count += 1

            try:
                # future.result() 返回 (doc_index, extracted_list)
                _, extracted_list = future.result()
                if extracted_list:
                    all_extracted_contradictions.extend(extracted_list)
                    successful_count += 1
                    print(f"✅ Index {doc_index} 提取成功。进度: {processed_count}/{doc_count}")
                else:
                    print(f"➖ Index {doc_index} 未找到矛盾或处理失败。进度: {processed_count}/{doc_count}")

            except Exception as exc:
                print(f'🚨 Index {doc_index} 任务生成了一个未捕获的异常: {exc}')

    print(f"\n--- 📊 统计 ---")
    print(f"   - 总文档数: {doc_count}")
    print(f"   - 成功提取文档数: {successful_count}")
    print(f"   - 总提取矛盾对数: {len(all_extracted_contradictions)}")

    return all_extracted_contradictions


# -------------------------
# 保存函数 (保持不变)
# -------------------------
def save_json_data(data: List[Dict[str, Any]], filename: str):
    """保存 JSON 格式的数据到指定文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"💾 Saved {len(data)} contradiction entries to {filename}")
    except Exception as e:
        print(f"⚠️ 错误：保存文件 {filename} 失败: {e}")


# -------------------------
# 主函数
# -------------------------
def main():
    # 假设输入文件路径
    input_file_path = "../../sampled_data.json"
    output_file_path = "dense_text.json"

    print("\n--- 🧠 任务: 使用 LLM 动态并发提取自相矛盾句子 (新格式) ---")

    # 1. 记录开始时间
    start_time = time.perf_counter()

    # 2. 执行核心提取任务
    llm_extracted_contradictions = extract_llm_contradictions_concurrent_new_format(
        input_file=input_file_path,
        llm=llm,
        system_prompt=SYSTEM_PROMPT
    )

    # 3. 如果有结果则保存
    if llm_extracted_contradictions:
        save_json_data(llm_extracted_contradictions, output_file_path)

    # 4. 记录结束时间并计算耗时
    end_time = time.perf_counter()
    duration = end_time - start_time

    # 5. 打印汇总报告
    print("\n" + "=" * 30)
    print("      ⏰ 运行时间统计")
    print("=" * 30)
    print(f"总执行耗时: {duration:.2f} 秒")

    # 如果处理了数据，计算单篇平均耗时
    # 我们可以通过读取 input_file_path 的长度来获取文档总数
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            total_docs = len(json.load(f))
        if total_docs > 0:
            print(f"平均处理速度: {duration / total_docs:.2f} 秒/文档")
    except:
        pass

    print(f"任务结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=" * 30)


if __name__ == "__main__":
    main()