"""
==================================================
📊 运行性能统计报告
--------------------------------------------------
📁 规则读取阶段耗时:   0.00 秒
🤖 并发 LLM 处理耗时:  13.13 秒
⏱️ 总程序运行耗时:     13.13 秒
📈 平均每个文件处理:   3.28 秒
✅ 处理完成时间:       2025-12-23 21:23:46
==================================================
"""
import os
import json
import re
import asyncio
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple
import time

# 设定目录
INPUT_DIR = "dual_star/llm_evaluations_v3/initial"
OUTPUT_DIR = "dual_star/llm_evaluations_v3/effective"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化环境
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=3000
)


# -------------------- I. 异步包装器 --------------------

async def _llm_generate_async(llm: OpenAIClient, user_message: str, system_message: str) -> str:
    """使用 asyncio.to_thread 包装同步的 LLM 调用。"""
    try:
        return await asyncio.to_thread(
            llm.generate,
            user_message=user_message,
            system_message=system_message
        )
    except Exception as e:
        print(f"❌ 警告: LLM 生成请求失败 (返回 None): {e}")
        return ""


# -------------------- II. 同步辅助函数 (保持不变或微调) --------------------

def read_contradictions_from_file(file_path: str) -> List[Dict[str, str]]:
    """
    从指定文件读取矛盾描述（同步）。由于每次只读取一个文件，保留列表格式以兼容后续步骤。
    """
    contradictions = []
    file_name = os.path.basename(file_path)

    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

            # 尝试解析规则和解释。这里的输入文件格式与原始代码不同，
            # 原始代码假设整个文件内容是一个矛盾reason，这不合理。
            # 我们假设输入文件是上一阶段保存的文本格式，包含 Rule: 和 Contradiction Explanation:

            # 为了简化和兼容，我们假设 Contradiction Explanation 的第一段文字是矛盾的 "reason"
            # 并且我们只处理找到的第一个 Contradiction Explanation

            reason_match = re.search(r'Contradiction Explanation:\s*(.*?)(?=\n\n|\Z)', content, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else content[:200] + "..."  # 截断作为reason

            # 将每个文件内容作为一个矛盾描述
            contradiction = {
                'file_name': file_name,
                'reason': reason
            }
            contradictions.append(contradiction)

    except Exception as e:
        print(f"⚠️ 警告: 读取或解析文件 {file_name} 失败: {e}")

    return contradictions


def read_rules_from_files(input_dir: str) -> Dict[str, List[str]]:
    """
    读取指定文件夹下的规则文件，并返回规则列表（同步）。
    由于输入文件现在是 `_effective_with_contradictions.txt` 格式，
    我们只提取其中的 `Rule:` 行。
    """
    rules_by_file = {}
    rule_pattern = re.compile(r'^Rule:\s*(.*)', re.MULTILINE)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                    # 查找所有以 "Rule:" 开头的行
                    rules = [match.group(1).strip() for match in rule_pattern.finditer(content)]
                    rules_by_file[file_name] = rules

            except Exception as e:
                print(f"⚠️ 警告: 读取文件 {file_name} 时出错: {e}")

    return rules_by_file


# -------------------- III. 异步核心逻辑 --------------------

async def extract_contradiction_positions_async(contradiction: Dict[str, str], llm: OpenAIClient) -> str:
    """
    异步提取单个矛盾的位置。
    """
    prompt = f"For the contradiction reason below, please identify the location or specific aspect of the contradiction (e.g., 'Numeric', 'Temporal', 'Emotional', 'Relational', etc.). Output ONLY the classification.\n\nContradiction: {contradiction['reason']}"

    system_message = "You are an expert in contradiction analysis. Your task is to extract the specific location or type of contradiction from the provided reasons. Output ONLY the classification."

    response = await _llm_generate_async(llm, user_message=prompt, system_message=system_message)

    # 清理响应，只保留第一个类型
    return response.strip().split('\n')[0].strip() if response else "LLM_Extraction_Failed"


async def refine_contradiction_positions_with_rules_async(contradiction: Dict[str, str], rules: List[str], llm: OpenAIClient) -> Tuple[str, List[str]]:
    """
    异步润色矛盾位置，并判断哪些规则反映矛盾。
    """

    prompt = f"""Refine the contradiction position for the following contradiction, considering the rules provided.

Contradiction: {contradiction['reason']}
Initial Position: {contradiction.get('contradiction_position', 'N/A')}

Rules to Evaluate:
{chr(10).join([f"- {rule}" for rule in rules])}

Your output must strictly follow this two-part format:
Refined Position: <The most specific contradiction type/aspect based on the rules and contradiction reason>
Matching Rules: <List the exact rules (one per line) from the 'Rules to Evaluate' section that logically reflect or predict this contradiction>
"""

    system_message = "You are an expert in contradiction analysis. Refine the contradiction position and identify the rules that logically reflect the contradiction. Ensure your output strictly adheres to the requested two-part format."

    response = await _llm_generate_async(llm, user_message=prompt, system_message=system_message)

    if not response:
        return contradiction.get('contradiction_position', 'LLM_Refinement_Failed'), []

    # 解析 LLM 的返回
    refined_position_match = re.search(r'Refined Position:\s*(.*)', response, re.IGNORECASE)
    matching_rules_section = re.search(r'Matching Rules:\s*(.*)', response, re.DOTALL | re.IGNORECASE)

    refined_position = refined_position_match.group(1).strip() if refined_position_match else contradiction.get('contradiction_position', 'LLM_Refinement_Failed')

    reflecting_rules = []
    if matching_rules_section:
        # 将规则列表按行分割和清理
        rules_text = matching_rules_section.group(1).strip()
        reflecting_rules = [line.strip() for line in rules_text.split('\n') if line.strip()]

    return refined_position, reflecting_rules


# -------------------- IV. 并发主处理函数 --------------------

async def process_single_file_concurrently(file_name: str, input_dir: str, output_dir: str, rules_by_file: Dict[str, List[str]], llm: OpenAIClient):
    """
    异步处理单个文件的所有逻辑：读取、提取位置、润色、匹配和保存。
    """
    file_path = os.path.join(input_dir, file_name)
    output_file_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_refined.json")

    print(f"🚀 [START] Processing {file_name}")

    # 1. 读取矛盾数据 (同步)
    contradictions = read_contradictions_from_file(file_path)

    if not contradictions:
        print(f"⚠️ [SKIP] {file_name}: No contradictions found or read error.")
        return

    # 由于 read_contradictions_from_file 只返回一个包含一个元素的列表，我们只处理第一个
    contradiction = contradictions[0]

    # 获取相关规则
    rules = rules_by_file.get(file_name, [])

    # 2. 提取矛盾位置 (异步 LLM 调用)
    initial_position = await extract_contradiction_positions_async(contradiction, llm)
    contradiction['contradiction_position'] = initial_position

    print(f"   [Step 2] Initial position for {file_name}: {initial_position}")

    # 3. 润色矛盾位置和匹配规则 (异步 LLM 调用)
    if rules:
        refined_position, reflecting_rules = await refine_contradiction_positions_with_rules_async(contradiction, rules, llm)

        # 更新矛盾数据
        contradiction['refined_position'] = refined_position
        contradiction['reflecting_rules'] = reflecting_rules
        print(f"   [Step 3] Refined position: {refined_position}, Found {len(reflecting_rules)} matching rules.")
    else:
        # 如果没有规则，则 refined_position 等于 initial_position
        contradiction['refined_position'] = initial_position
        contradiction['reflecting_rules'] = []
        print(f"   [Step 3] No rules found. Skipping refinement.")

    # 4. 保存数据 (同步)
    try:
        # 将更新后的数据保存到文件中
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump([contradiction], f, ensure_ascii=False, indent=2)
        print(f"✅ [SUCCESS] Data saved to {os.path.basename(output_file_path)}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to save data for {file_name}: {e}")


# -------------------- V. 异步主函数 --------------------

async def main_concurrent():
    # 1. 记录程序启动的绝对时间
    overall_start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] --- 规则评估和润色流程开始 (并发模式) ---")

    # 2. 读取所有规则文件 (同步操作计时)
    io_start = time.time()
    rules_by_file = read_rules_from_files(INPUT_DIR)
    io_duration = time.time() - io_start

    if not rules_by_file:
        print("🔴 无法继续：在输入目录中未找到任何规则文件。")
        return

    # 3. 准备并发任务列表
    tasks = []
    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith('.txt') and file_name in rules_by_file:
            task = process_single_file_concurrently(
                file_name=file_name,
                input_dir=INPUT_DIR,
                output_dir=OUTPUT_DIR,
                rules_by_file=rules_by_file,
                llm=llm
            )
            tasks.append(task)

    if not tasks:
        print("⚠️ 警告: 没有找到匹配的输入文件来处理。")
        return

    # 4. 并发执行阶段计时
    print(f"--- 🚀 发现 {len(tasks)} 个文件需要处理，启动并发 LLM 调用 ---")
    llm_start = time.time()

    await asyncio.gather(*tasks)

    llm_duration = time.time() - llm_start

    # 5. 计算总耗时并打印统计报告
    total_elapsed = time.time() - overall_start_time

    print("\n" + "=" * 50)
    print(f"📊 运行性能统计报告")
    print("-" * 50)
    print(f"📁 规则读取阶段耗时:   {io_duration:.2f} 秒")
    print(f"🤖 并发 LLM 处理耗时:  {llm_duration:.2f} 秒")
    print(f"⏱️ 总程序运行耗时:     {total_elapsed:.2f} 秒")
    if len(tasks) > 0:
        print(f"📈 平均每个文件处理:   {total_elapsed / len(tasks):.2f} 秒")
    print(f"✅ 处理完成时间:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)


if __name__ == "__main__":
    # 使用 asyncio.run 启动
    asyncio.run(main_concurrent())