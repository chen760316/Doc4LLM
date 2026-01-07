"""
=============================================
⏱️  运行时间统计报告
---------------------------------------------
⌛ 总运行耗时: 45.67 秒
📑 平均每文档耗时: 4.57 秒
🚀 处理吞吐量: 0.22 docs/sec
---------------------------------------------
✅ 成功提取矛盾对: 0
=============================================
"""
import os
import json
import time  # 1. 引入 time 模块
import re
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# 配置
# -------------------------
MAX_WORKERS = 10
MAX_RETRIES = 3

# ... [环境变量加载与初始化保持不变] ...
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.2,
    top_p=0.9,
    max_tokens=4000
)

# ... [SYSTEM_PROMPT 保持不变] ...
SYSTEM_PROMPT = """
You are an expert in analyzing documents to detect contradictions.
... (内容省略) ...
"""


# ... [process_single_entry_new_format 函数保持不变] ...
def process_single_entry_new_format(doc_index: int, entry: Dict[str, Any], llm: OpenAIClient, system_prompt: str) -> Tuple[int, List[Dict[str, Any]]]:
    conflicting_texts = entry.get("complex_text", [])
    full_text = "\n".join(conflicting_texts).strip() if isinstance(conflicting_texts, list) else str(conflicting_texts).strip()

    if not full_text:
        return doc_index, []

    user_prompt = f"Analyze the following passage and find all pairs of contradicting sentences within it:\n\nPassage:\n{full_text}"

    retry_count = 0
    success = False
    result_str = ""

    while retry_count < MAX_RETRIES and not success:
        try:
            resp = llm.generate(user_message=user_prompt, system_message=system_prompt)
            result_str = resp.strip()
            if result_str == "No Contradiction Found.":
                success = True
                continue

            # 清理 JSON 围栏
            clean_json = re.sub(r'```json\s*|\s*```', '', result_str).strip()
            contradictions = json.loads(clean_json)
            success = True
        except Exception:
            retry_count += 1
            time.sleep(1)

    if success and result_str != "No Contradiction Found." and isinstance(contradictions, list):
        final_results = []
        for item in contradictions:
            item["source_index"] = doc_index
            item["source_main_entity"] = entry.get("main_entity", "N/A")
            item["source_domain"] = entry.get("source_domain", "N/A")
            item["extraction_method"] = "LLM_Self_Contradiction"
            s1 = item.pop('sentence_1', '')
            s2 = item.pop('sentence_2', '')
            item["text"] = f"{s1} [SEP] {s2}"
            final_results.append(item)
        return doc_index, final_results
    return doc_index, []


# ... [extract_llm_contradictions_concurrent_new_format 保持不变] ...

def save_json_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
# 主函数 (添加计时逻辑)
# -------------------------
def main():
    input_file_path = "../../sampled_data.json"
    output_file_path = "dense_text.json"

    print("\n--- 🧠 任务: 使用 LLM 动态并发提取自相矛盾句子 (新格式) ---")

    # 2. 记录起始时间
    start_time = time.time()

    # 执行核心提取逻辑
    llm_extracted_contradictions = extract_llm_contradictions_concurrent_new_format(
        input_file=input_file_path,
        llm=llm,
        system_prompt=SYSTEM_PROMPT
    )

    # 3. 记录结束时间并计算耗时
    end_time = time.time()
    total_duration = end_time - start_time

    # 4. 获取文档总数用于计算平均值
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            total_docs = len(json.load(f))
    except:
        total_docs = 0

    # 5. 打印运行时间报告
    print("\n" + "=" * 45)
    print("⏱️  运行时间统计报告")
    print("-" * 45)
    print(f"⌛ 总运行耗时: {total_duration:.2f} 秒")
    if total_docs > 0:
        avg_time = total_duration / total_docs
        tps = total_docs / total_duration
        print(f"📑 平均每文档耗时: {avg_time:.2f} 秒")
        print(f"🚀 处理吞吐量: {tps:.2f} docs/sec")
    print("-" * 45)
    print(f"✅ 成功提取矛盾对: {len(llm_extracted_contradictions)}")
    print("=" * 45)

    if llm_extracted_contradictions:
        save_json_data(llm_extracted_contradictions, output_file_path)
        print(f"💾 结果已保存到 {output_file_path}")


if __name__ == "__main__":
    main()