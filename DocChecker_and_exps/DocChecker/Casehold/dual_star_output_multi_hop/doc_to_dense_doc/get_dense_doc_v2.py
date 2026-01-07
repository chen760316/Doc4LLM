import os
import json
import time
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed # 引入多线程模块

# -------------------------
# 配置
# -------------------------
MAX_WORKERS = 12 # 线程池最大工作线程数
MAX_RETRIES = 3 # LLM调用最大重试次数

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
# 单个文档处理函数 (适配并发)
# -------------------------
def process_single_entry(
    unique_id: str,
    entry: Dict[str, Any],
    llm: OpenAIClient,
    system_prompt: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    处理单个文档条目，调用 LLM 提取矛盾，并执行重试和格式清理。
    返回 (unique_id, 提取的矛盾列表)。
    """
    full_text = entry.get("text", "")
    doc_type = entry.get("doc_type", "N/A")
    if not full_text:
        return unique_id, []

    user_prompt = f"Analyze the following passage and find all pairs of contradicting sentences within it:\n\nPassage:\n{full_text}"
    # print(f"Calling LLM for ID: {unique_id}...") # 仅在重试时打印，避免并发时的输出混乱

    # --- 重试机制 ---
    retry_count = 0
    success = False
    result_str = ""

    while retry_count < MAX_RETRIES and not success:
        try:
            if retry_count > 0:
                print(f"   重试 ({retry_count}/{MAX_RETRIES - 1}): ID {unique_id}")
                time.sleep(2 ** retry_count)  # 退避等待

            resp = llm.generate(user_message=user_prompt, system_message=system_prompt)
            result_str = resp.strip()

            if result_str == "No Contradiction Found.":
                success = True
                continue

            # --- 🎯 格式清理核心 ---
            if result_str.startswith("```json"):
                result_str = result_str[7:]
            if result_str.endswith("```"):
                result_str = result_str[:-3]
            result_str = result_str.strip()
            # -------------------------

            # 尝试解析 JSON
            contradictions = json.loads(result_str)
            success = True  # 解析成功，退出循环

        except json.JSONDecodeError:
            # print(f"   JSON 解析失败 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
            retry_count += 1
        except Exception as e:
            # 捕获其他异常，如网络错误或 LLM API 错误
            # print(f"   LLM 调用失败 (尝试 {retry_count + 1}/{MAX_RETRIES}): {e}")
            retry_count += 1

    # --- 结果处理 ---
    if not success:
        print(f"❌ 达到最大重试次数，ID {unique_id} 处理失败。")
        return unique_id, []

    if result_str == "No Contradiction Found.":
        return unique_id, []

    # 确保 contradictions 变量在成功时被设置
    if success and 'contradictions' in locals() and isinstance(contradictions, list):
        final_results = []
        for item in contradictions:
            item["source_id"] = unique_id
            item["extraction_method"] = "LLM_Self_Contradiction"

            # 🎯 核心修改点：合并 sentence_1 和 sentence_2 到 'text' 字段
            sentence_1 = item.pop('sentence_1', '')
            sentence_2 = item.pop('sentence_2', '')
            item["text"] = f"{sentence_1} [SEP] {sentence_2}"

            # 保留其他元数据
            item["doc_type"] = doc_type

            final_results.append(item)

        return unique_id, final_results

    return unique_id, []


# -------------------------
# LLM 动态提取函数 (并发实现)
# -------------------------
def extract_llm_contradictions_concurrent(input_file: str, llm: OpenAIClient, system_prompt: str) -> List[Dict[str, Any]]:
    """
    使用 ThreadPoolExecutor 遍历 'pos' 字段，并发地提取自相矛盾的句子对。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ 错误：加载文件 {input_file} 失败: {e}")
        return []

    # 🎯 核心修改：只处理 'pos' 字段，并将其视为文档列表
    pos_entries = data.get("pos", {})
    all_extracted_contradictions = []

    doc_count = len(pos_entries)
    processed_count = 0
    successful_count = 0

    print(f"\n--- ⚙️ 启动线程池，文档总数: {doc_count}, 最大工作线程数: {MAX_WORKERS} ---")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_id = {
            executor.submit(
                process_single_entry, unique_id, entry, llm, system_prompt
            ): unique_id
            for unique_id, entry in pos_entries.items()
        }

        # 收集结果
        for future in as_completed(future_to_id):
            unique_id = future_to_id[future]
            processed_count += 1

            try:
                # future.result() 返回 (unique_id, extracted_list)
                _, extracted_list = future.result()
                if extracted_list:
                    all_extracted_contradictions.extend(extracted_list)
                    successful_count += 1
                    # print(f"✅ ID {unique_id} 提取成功。进度: {processed_count}/{doc_count}")
                # else:
                    # print(f"➖ ID {unique_id} 未找到矛盾或处理失败。进度: {processed_count}/{doc_count}")

            except Exception as exc:
                print(f'🚨 ID {unique_id} 任务生成了一个未捕获的异常: {exc}')

    print(f"\n--- 📊 统计 ---")
    print(f"   - 总文档数 (pos): {doc_count}")
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
    input_file_path = "../../sampled_data.json"
    output_file_path = "dense_text.json"

    print("\n--- 🧠 任务: 使用 LLM 动态并发提取自相矛盾句子 ---")

    # 1. 使用 LLM 动态提取矛盾
    llm_extracted_contradictions = extract_llm_contradictions_concurrent(
        input_file=input_file_path,
        llm=llm,
        system_prompt=SYSTEM_PROMPT
    )

    # 2. 统一保存结果
    if llm_extracted_contradictions:
        save_json_data(llm_extracted_contradictions, output_file_path)
        print("\n--- ✅ 任务完成 ---")
        print(f"✨ 所有 LLM 提取的自相矛盾句子对已合并并保存到 {output_file_path}。")
    else:
        print("\n--- ⚠️ 任务完成 ---")
        print("未提取到任何矛盾，未生成输出文件。")


if __name__ == "__main__":
    main()