import json
import os
import re
from glob import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple, Set as TypeSet

# 初始化环境 (保持不变)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 假设 OpenAIClient 已正确初始化
llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=3500
)

# 定义批处理大小 (LLM 调用次数和延迟的权衡)
CONTRADICTION_BATCH_SIZE = 150  # 保持不变，用于矛盾分类
RULES_BATCH_SIZE = 10  # 💥 优化点：规则评估批量减小到 10，以提高响应速度
MAX_RULES_PER_FILE = 10  # 💥 优化点：每个文件只评估前 10 条规则

# 全局线程池，用于异步执行所有同步的磁盘I/O操作
IO_EXECUTOR = ThreadPoolExecutor(max_workers=5)


# -------------------- I. LLM 核心函数重构 (I/O & LLM 逻辑) --------------------

# 1. 异步加载矛盾数据 (保持不变)
async def load_contradictions_async(file_path: str) -> List[Dict[str, Any]]:
    """异步加载 JSON 文件以避免阻塞主线程"""

    def _load():
        # 同步文件加载逻辑
        if not file_path.endswith('.json'):
            print(f"⚠️ 警告: 正在尝试加载非 JSON 文件 {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return await asyncio.get_event_loop().run_in_executor(IO_EXECUTOR, _load)


# 2. 批量提取矛盾位置 (保持不变)
async def extract_contradiction_positions_batched(contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> Dict[int, str]:
    """
    将矛盾原因批处理，返回一个全局索引到分类结果的字典。
    """

    # 包装同步的 LLM 批量调用
    async def _batch_call(batch_index_start: int, batch_reasons: List[str]):
        """执行单个 LLM 批处理请求，并明确在 Prompt 中要求返回全局索引"""
        prompt = "Contradictions to classify:\n" + "\n".join(batch_reasons)

        response = await asyncio.to_thread(
            llm.generate,
            user_message=prompt,
            # 明确要求返回全局索引 (Contradiction X:)，提高解析准确性
            system_message=f"You are an expert in contradiction analysis. Your task is to extract the specific location or type of contradiction from the provided reasons. For each numbered contradiction, output ONLY the extracted type on a new line, prefixed by the corresponding 'Contradiction X: ' label (e.g., 'Contradiction {batch_index_start}: Factual')."
        )
        return response

    # 创建全局索引的 Reason 列表
    reasons = [f"Contradiction {i}: {c.get('reason', 'No reason')}" for i, c in enumerate(contradictions)]

    # 创建批次，并记录每个批次的起始全局索引
    batches: List[Tuple[int, List[str]]] = []
    for i in range(0, len(reasons), CONTRADICTION_BATCH_SIZE):
        batches.append((i, reasons[i:i + CONTRADICTION_BATCH_SIZE]))

    print(f"🧠 正在批处理 {len(reasons)} 个矛盾原因，分为 {len(batches)} 个 LLM 请求...")

    # 并发执行批次
    tasks = [_batch_call(start_index, batch) for start_index, batch in batches]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_positions: Dict[int, str] = {}

    # 逐批解析并合并结果
    for result, (start_index, batch) in zip(batch_results, batches):
        if isinstance(result, Exception) or not result:
            print(f"⚠️ 警告: LLM 批处理请求失败，当前批次使用默认值。错误: {result}")
            # 为当前批次的所有索引设置默认值
            for i in range(len(batch)):
                all_positions[start_index + i] = "LLM_Classification_Failed"
            continue

        # 🌟 优化：使用 re.findall 查找所有匹配项，不依赖顺序
        # 匹配 Contradiction X: <类型>，X 是全局索引
        matches = re.findall(r'Contradiction\s+(\d+):\s*(.*)', result, re.IGNORECASE)

        for global_index_str, classification in matches:
            try:
                global_index = int(global_index_str)
                # 只有当全局索引在原始矛盾列表范围内时，才接受结果
                if 0 <= global_index < len(contradictions):
                    all_positions[global_index] = classification.strip() or "Missing_Classification"
            except ValueError:
                continue

    # 确保所有矛盾都有分类结果（LLM可能漏掉或格式错误）
    for i in range(len(contradictions)):
        if i not in all_positions:
            all_positions[i] = "Missing_Classification"

    return all_positions


# 3. 异步加载规则文件 (保持不变)
async def load_rules_from_llm_async(folder_path: str) -> Dict[str, List[str]]:
    """异步加载规则文件，以防止阻塞主线程"""

    def _load():
        rules_by_file = {}
        rule_start_pattern = re.compile(r'Rules:\s*')
        if not os.path.isdir(folder_path):
            print(f"🔴 错误: 规则文件夹未找到: {folder_path}")
            return {}

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                        match = rule_start_pattern.search(content)
                        if match:
                            rules_section = content[match.end():].strip()
                            rules_list = rules_section.split('\n')
                            rules_by_file[filename] = [rule.strip() for rule in rules_list if rule.strip()]
                except Exception as e:
                    print(f"⚠️ 警告: 读取文件 {filename} 时出错: {e}")
        return rules_by_file

    return await asyncio.get_event_loop().run_in_executor(IO_EXECUTOR, _load)


# 4. 异步批处理匹配规则到矛盾类型 (保持不变)
async def _match_rule_batch(rules_batch: List[str], simplified_contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> Dict[str, str]:
    """
    为一批规则生成一个 prompt，并返回它们的解释。
    """
    # 创建规则索引映射，方便后续根据索引匹配原始规则
    rule_map = {f"Rule {i + 1}": rule for i, rule in enumerate(rules_batch)}
    rule_list_prompt = "\n".join([f"Rule {i + 1}: {rule}" for i, rule in enumerate(rules_batch)])

    prompt = f"""Evaluate the following rules against the given contradictions. The goal is to determine if the rule's premise and conclusion logically align with the patterns of the contradictions provided.

All Contradictions (Simplified):
{json.dumps(simplified_contradictions, ensure_ascii=False, indent=2)}

Rules to Evaluate:
{rule_list_prompt}

For each rule, return the explanation in the format 'Rule X: <Explanation>' only for the rules that **logically align** with a contradiction pattern. **DO NOT include rules that do not align or need no explanation.**
"""

    response = await asyncio.to_thread(
        llm.generate,
        user_message=prompt,
        system_message="You are an expert in logical rule evaluation. Only output the explanations in the exact format: 'Rule X: <Explanation>' for rules that align with the contradiction patterns."
    )

    rule_match_results: Dict[str, str] = {}
    if not response:
        return {}

    # 🌟 优化：使用 re.findall 查找所有匹配项
    # 匹配 Rule X: <Explanation>
    matches = re.findall(r'Rule\s+(\d+):\s*(.*)', response, re.DOTALL | re.IGNORECASE)

    for index_str, explanation in matches:
        try:
            # 批次内索引 (从 1 开始)
            index = int(index_str)
            # 找到原始规则
            original_rule = rule_map.get(f"Rule {index}")

            if original_rule:
                # 只有在 LLM 提供了非空的解释时，才认为是匹配成功的
                cleaned_explanation = explanation.strip()
                if cleaned_explanation:
                    rule_match_results[original_rule] = cleaned_explanation
        except ValueError:
            continue

    return rule_match_results


# 4. 并发匹配规则的主函数 (保持不变)
async def match_rules_to_contradictions_concurrent(rules: List[str], contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> Dict[str, str]:
    # 简化矛盾数据 (只在内存中创建一次)
    simplified_contradictions = [{"id": i, "type": c.get('contradiction_type'), "position": c.get('contradiction_position')} for i, c in enumerate(contradictions)]

    # 创建规则批次
    batches = [rules[i:i + RULES_BATCH_SIZE] for i in range(0, len(rules), RULES_BATCH_SIZE)]

    print(f"🧠 正在批处理 {len(rules)} 条规则，分为 {len(batches)} 个 LLM 请求...")

    tasks = [_match_rule_batch(batch, simplified_contradictions, llm) for batch in batches]

    # 并发执行批次
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    rule_match_results: Dict[str, str] = {}

    # 合并结果
    for result in batch_results:
        if isinstance(result, Exception):
            print(f"⚠️ 警告: 规则批处理请求失败，忽略此结果。错误: {result}")
        else:
            # _match_rule_batch 只返回了成功匹配的规则，直接更新
            rule_match_results.update(result)

    # 最终结果只包含成功匹配并有解释的规则
    return rule_match_results


# 5. 异步保存有效规则 (保持不变)
async def save_matched_rules_async(effective_rules: Dict[str, List[Tuple[str, str]]], output_folder: str):
    """异步保存有效规则，以防止阻塞主线程"""

    def _save():
        os.makedirs(output_folder, exist_ok=True)

        for filename, rules_data in effective_rules.items():
            sub_folder = os.path.join(output_folder, "initial")
            os.makedirs(sub_folder, exist_ok=True)

            # 修正文件名，确保后缀正确
            base_name = os.path.splitext(filename)[0].replace('_rules_llm', '')
            output_path = os.path.join(sub_folder, f"{base_name}_evaluated.txt")

            with open(output_path, "w", encoding="utf-8") as f:
                for rule, explanation_string in rules_data:
                    # 在优化后的逻辑中，传入的 rules_data 只包含成功匹配的规则
                    if explanation_string:
                        f.write(f"Rule: {rule}\n")
                        f.write(f"Contradiction Explanation: {explanation_string}\n\n")

        print(f"\n✅ Successfully saved effective rules with contradiction explanations in: {output_folder}")

    await asyncio.get_event_loop().run_in_executor(IO_EXECUTOR, _save)


# -------------------- II. 主程序重构为异步 (核心修改在这里) --------------------

async def main():
    contradictions_file = 'dense_text.json'
    llm_folder = './dual_star/llm_rules'
    output_folder = './dual_star/llm_evaluations_v3'

    print(f"--- 🚀 评估流程开始 (优化版：高并发 + 异步 I/O + 限制规则数) ---")

    # 1. 异步加载矛盾数据
    try:
        contradictions = await load_contradictions_async(contradictions_file)
        print(f"Loaded {len(contradictions)} contradictions.")
    except Exception as e:
        print(f"🔴 错误: 异步加载矛盾文件失败，请确保文件 {contradictions_file} 存在且是有效的 JSON 格式: {e}")
        return

    # 2. 通过 LLM 批量提取矛盾位置
    positions_map = await extract_contradiction_positions_batched(contradictions, llm)

    # 将结果合并回矛盾列表
    contradictions_with_positions = []
    for i, c in enumerate(contradictions):
        c['contradiction_position'] = positions_map.get(i, "Missing_Classification")
        contradictions_with_positions.append(c)

    # 3. 异步加载规则文件
    rules_by_file = await load_rules_from_llm_async(llm_folder)

    if not rules_by_file:
        print("🔴 无法继续：没有找到任何规则文件。")
        return

    effective_rules: Dict[str, List[Tuple[str, str]]] = {}
    all_tasks = []

    # 4. 异步并行处理每个文件中的规则评估 (应用 MAX_RULES_PER_FILE 限制)
    for filename, all_rules in rules_by_file.items():
        # 💥 核心修改：只评估每个文件的前 MAX_RULES_PER_FILE 条规则
        rules_to_evaluate = all_rules[:MAX_RULES_PER_FILE]

        if not rules_to_evaluate:
            print(f"⚠️ 文件 {filename} 中没有规则，跳过评估。")
            continue

        async def evaluate_file(fn, rls, cont_w_pos):
            print(f"\n--- ⏳ 正在评估文件: {fn} ({len(rls)} 条规则, 原始总数: {len(all_rules)}) ---")

            # 使用 LLM 批量并发判断规则与矛盾的匹配 (只返回匹配成功的规则)
            matching_results = await match_rules_to_contradictions_concurrent(rls, cont_w_pos, llm)

            # 整理文件规则数据：只保留匹配成功的规则及其解释
            file_rules_data = []
            for rule in rls:
                explanation_string = matching_results.get(rule)
                if explanation_string:  # 只有当匹配成功且有解释时才加入
                    file_rules_data.append((rule, explanation_string))

            return fn, file_rules_data

        all_tasks.append(evaluate_file(filename, rules_to_evaluate, contradictions_with_positions))

    # 并发执行所有文件的评估任务
    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # 合并结果
    for result in results:
        if isinstance(result, Exception):
            print(f"⚠️ 警告: 文件评估任务失败。错误: {result}")
        else:
            filename, file_rules_data = result
            effective_rules[filename] = file_rules_data

    # 5. 异步保存有效规则
    await save_matched_rules_async(effective_rules, output_folder)

    # 关闭全局线程池
    IO_EXECUTOR.shutdown(wait=False)
    print("--- ✅ 评估流程完成 ---")


if __name__ == '__main__':
    # 使用 asyncio.run() 来运行顶层的异步 main 函数
    asyncio.run(main())