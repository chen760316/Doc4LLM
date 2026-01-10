import os
import json
import time
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
from typing import Dict, Any, List

# -------------------------
# 环境变量加载 (保持不变)
# -------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# -------------------------
# 初始化 LLM (保持不变)
# -------------------------
llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.2,
    top_p=0.9,
    max_tokens=4000
)

# -------------------------
# 系统提示词 (用于LLM提取)
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


# Note: I've simplified the system prompt here, assuming the full version provided previously
# (with all examples and guidelines) will be used in the actual execution.

# -------------------------
# LLM 动态提取函数 (核心修改)
# -------------------------
def extract_llm_contradictions(input_file: str, llm: OpenAIClient, system_prompt: str) -> List[Dict[str, Any]]:
    """
    遍历 sampled_contradoc.json 的 'pos' 字段，将每个 entry 的完整 'text'
    字段输入给 LLM，让 LLM 在单个文本中动态提取自相矛盾的句子对。
    加入了格式清理和重试机制 (最多 3 次)。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ 错误：加载文件 {input_file} 失败: {e}")
        return []

    pos_entries = data.get("pos", {})
    all_llm_results = []

    for unique_id, entry in pos_entries.items():
        full_text = entry.get("text", "")
        if not full_text:
            continue

        user_prompt = f"Analyze the following passage and find all pairs of contradicting sentences within it:\n\nPassage:\n{full_text}"
        print(f"Calling LLM for ID: {unique_id}...")

        # --- 重试机制初始化 ---
        max_retries = 3
        retry_count = 0
        success = False
        result_str = ""

        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    print(f"   重试 ({retry_count}/{max_retries - 1}): ID {unique_id}")
                    time.sleep(2 ** retry_count)  # 退避等待 2s, 4s

                resp = llm.generate(user_message=user_prompt, system_message=system_prompt)
                result_str = resp.strip()

                if result_str == "No Contradiction Found.":
                    success = True
                    continue

                # --- 🎯 格式清理核心 ---
                # 移除常见的 Markdown JSON 围栏
                if result_str.startswith("```json"):
                    result_str = result_str[7:]
                if result_str.endswith("```"):
                    result_str = result_str[:-3]

                # 移除 JSON 围栏后的前后空白
                result_str = result_str.strip()
                # -------------------------

                # 尝试解析 JSON
                contradictions = json.loads(result_str)
                success = True  # 解析成功，退出循环

            except json.JSONDecodeError as e:
                print(f"   JSON 解析失败 (尝试 {retry_count + 1}/{max_retries}): {e}")
                print(f"   LLM 原始输出 (用于调试): {result_str[:200]}...")
                retry_count += 1

            except Exception as e:
                # 捕获其他异常，如网络错误或 LLM API 错误
                print(f"   LLM 调用失败 (尝试 {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1

        # --- 结果处理 ---
        if not success:
            print(f"❌ 达到最大重试次数，ID {unique_id} 处理失败。")
            continue

        if result_str == "No Contradiction Found.":
            continue

        # 确保 contradictions 变量在成功时被设置
        if success and 'contradictions' in locals():
            for item in contradictions:
                # 标记 LLM 提取的来源
                item["source_id"] = unique_id
                item["extraction_method"] = "LLM_Self_Contradiction"

                # 🎯 核心修改点：合并 sentence_1 和 sentence_2 到 'text' 字段
                sentence_1 = item.pop('sentence_1', '')
                sentence_2 = item.pop('sentence_2', '')
                item["text"] = f"{sentence_1} [SEP] {sentence_2}"

                # 保留其他元数据
                item["doc_type"] = entry.get("doc_type", "N/A")

                all_llm_results.append(item)

    return all_llm_results

# -------------------------
# 保存函数
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
    input_file_path = "../../sampled_contradoc.json"
    output_file_path = "dense_text.json"  # 统一保存到这一个 JSON 文件

    print("\n--- 🧠 任务: 使用 LLM 动态提取自相矛盾句子 ---")

    # 1. 使用 LLM 动态提取矛盾
    llm_extracted_contradictions = extract_llm_contradictions(
        input_file=input_file_path,
        llm=llm,
        system_prompt=SYSTEM_PROMPT  # 使用修改后的 SYSTEM_PROMPT
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