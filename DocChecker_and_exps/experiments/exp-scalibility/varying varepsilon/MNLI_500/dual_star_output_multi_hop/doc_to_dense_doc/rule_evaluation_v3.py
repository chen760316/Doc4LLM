"""
==================================================
           ✨ 批量评估任务执行报告 ✨
==================================================
📊 数据规模:
   - 矛盾条数: 2012
   - 规则总数: 2691
   - 处理文件: 102 个
--------------------------------------------------
⏱️ 耗时明细:
   - 阶段 1 (矛盾位置分类): 26.30 秒
   - 阶段 2 (规则逻辑匹配): 1255.44 秒
   - 🚀 总执行耗时: 1281.98 秒 (约 21.37 分钟)
--------------------------------------------------
📈 吞吐量指标:
   - 规则处理速度: 2.10 条/秒
   - 每分钟处理量: 220.11 任务项/分钟
==================================================
📅 结束时间: 2025-12-24 21:16:12
==================================================
"""
import json
import os
import re
from glob import glob
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple, Set as TypeSet
import time

# 初始化环境 (保持不变)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 假设 OpenAIClient 已正确初始化
llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=3500  # 增加 max_tokens 以适应批量处理
)

# 定义批处理大小
CONTRADICTION_BATCH_SIZE = 100  # 矛盾分类，可一次处理大量
RULES_BATCH_SIZE = 15  # 规则评估，需要同时发送矛盾数据，因此批次不能太大


# -------------------- I. LLM 核心函数重构 --------------------

# 1. 读取矛盾数据 (保持不变)
def load_contradictions(file_path: str) -> List[Dict[str, Any]]:
    # 路径修正，假设用户输入的是 dense_text.json，如果不是，需要调整
    if not file_path.endswith('.json'):
        # 假设文件名为 dense_text.txt，内容是 JSON 数组或每行一个 JSON 对象
        # 这里为了保持 load_contradictions 的通用性，我们假设它加载的是一个 JSON 数组
        # 如果文件是 .txt 但内容是 JSON array，则无需修改
        print(f"⚠️ 警告: 正在尝试加载非 JSON 文件 {file_path}，请确保其内容是有效的 JSON 数组。")

    with open(file_path, 'r', encoding='utf-8') as f:
        # 注意: load_contradictions 应该返回 List[Dict]
        return json.load(f)


# 2. 批量提取矛盾位置
async def extract_contradiction_positions_batched(contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> List[str]:
    """
    将所有矛盾的原因批处理到一个 LLM 请求中，以减少 API 调用次数。
    返回一个与输入矛盾列表长度相同的“位置”列表。
    """
    reasons = [f"Contradiction {i}: {c.get('reason', 'No reason')}" for i, c in enumerate(contradictions)]

    # 包装同步的 LLM 批量调用
    async def _batch_call(batch_reasons: List[str]):
        prompt = "Contradictions to classify:\n" + "\n".join(batch_reasons)

        response = await asyncio.to_thread(
            llm.generate,
            user_message=prompt,
            system_message="You are an expert in contradiction analysis. Your task is to extract the specific location or type of contradiction from the provided reasons. For each numbered contradiction, output ONLY the extracted type on a new line, prefixed by the corresponding 'Contradiction X: ' label (e.g., 'Contradiction 1: Factual', 'Contradiction 2: Numeric')."
        )
        return response

    # 如果矛盾数量小于批次大小，则一次性处理
    if len(reasons) <= CONTRADICTION_BATCH_SIZE:
        batches = [reasons]
    else:
        # 创建批次
        batches = [reasons[i:i + CONTRADICTION_BATCH_SIZE] for i in range(0, len(reasons), CONTRADICTION_BATCH_SIZE)]

    print(f"🧠 正在批处理 {len(reasons)} 个矛盾原因，分为 {len(batches)} 个 LLM 请求...")

    # 并发执行批次
    batch_results = await asyncio.gather(*[_batch_call(batch) for batch in batches], return_exceptions=True)

    all_positions: List[str] = ["LLM_Classification_Failed"] * len(contradictions)

    current_index = 0
    # 解析并合并结果
    for result in batch_results:
        if isinstance(result, Exception) or not result:
            print(f"⚠️ 警告: LLM 批处理请求失败，当前批次使用默认值。错误: {result}")
            current_index += len(batches[batch_results.index(result)])
            continue

        # 逐行解析 LLM 响应
        for line in result.strip().split('\n'):
            line = line.strip()
            # 查找 Contradiction X: 类型
            match = re.match(r'Contradiction\s+(\d+):\s*(.*)', line, re.IGNORECASE)
            if match:
                # 提取数字索引和分类结果
                # 由于索引是全局的，我们需要重新计算它相对于原始列表的位置
                # 注意：这里需要依赖 LLM 正确返回 Contradiction X: 格式，且 X 应该从 0 开始

                # 简单方法：依赖于 LLM 严格按顺序返回
                if current_index < len(all_positions):
                    # 尝试清理和存储分类结果
                    classification = match.group(2).strip()
                    all_positions[current_index] = classification if classification else "Missing_Classification"
                    current_index += 1
            else:
                # 如果 LLM 输出格式不规范，直接依赖顺序，但这是不稳健的
                # 为了简化，我们继续递增索引，但这可能会导致结果错位
                if current_index < len(all_positions):
                    all_positions[current_index] = line if line else "Missing_Classification"
                    current_index += 1

    # 裁剪到实际矛盾数量，以防解析错误多加
    return all_positions[:len(contradictions)]


# 3. 加载规则文件并与矛盾进行匹配 (保持不变)
def load_rules_from_llm(folder_path: str) -> Dict[str, List[str]]:
    # ... (函数体保持不变)
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


# 4. 异步批处理匹配规则到矛盾类型
async def _match_rule_batch(rules_batch: List[str], simplified_contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> Dict[str, str]:
    """
    为一批规则生成一个 prompt，并返回它们的解释。
    """
    rule_list_prompt = "\n".join([f"Rule {i + 1}: {rule}" for i, rule in enumerate(rules_batch)])

    prompt = f"""Evaluate the following rules against the given contradictions. The goal is to determine if the rule's premise and conclusion logically align with the patterns of the contradictions provided.

All Contradictions (Simplified):
{json.dumps(simplified_contradictions, ensure_ascii=False)}

Rules to Evaluate:
{rule_list_prompt}

For each rule, return the explanation in the format 'Rule X: <Explanation>' only for the rules that align with a contradiction pattern. If a rule does not align or no explanation is needed, do not include it in the output.
"""

    response = await asyncio.to_thread(
        llm.generate,
        user_message=prompt,
        system_message="You are an expert in logical rule evaluation against contradiction patterns. Your task is to provide a concise explanation for each provided rule about whether it aligns with the contradiction patterns given in the 'All Contradictions' list. Output only the explanations, one per rule, in the exact format: 'Rule X: <Explanation>'"
    )

    rule_match_results: Dict[str, str] = {}
    if not response:
        return {}

    # 逐行解析 LLM 响应
    for i, rule in enumerate(rules_batch):
        # 预期格式：Rule X: <Explanation>
        # 由于我们无法保证 LLM 严格遵守格式且按顺序返回，我们使用正则表达式匹配
        match = re.search(rf'Rule\s+({i + 1}):\s*(.*)', response, re.DOTALL | re.IGNORECASE)

        # 如果找到匹配项，使用它；否则使用一个默认值
        explanation = match.group(2).strip() if match and match.group(2) else f"No detailed explanation provided by LLM for Rule {i + 1}."

        rule_match_results[rule] = explanation

    return rule_match_results


# 4. 并发匹配规则的主函数 (调用批处理函数)
async def match_rules_to_contradictions_concurrent(rules: List[str], contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> Dict[str, str]:
    # 简化矛盾数据 (只在内存中创建一次)
    simplified_contradictions = [{"id": i, "type": c.get('contradiction_type'), "position": c.get('contradiction_position')} for i, c in enumerate(contradictions)]

    # 创建规则批次
    batches = [rules[i:i + RULES_BATCH_SIZE] for i in range(0, len(rules), RULES_BATCH_SIZE)]

    print(f"🧠 正在批处理 {len(rules)} 条规则，分为 {len(batches)} 个 LLM 请求...")

    tasks = []
    for batch in batches:
        tasks.append(_match_rule_batch(batch, simplified_contradictions, llm))

    # 并发执行批次
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    rule_match_results: Dict[str, str] = {}

    # 合并结果
    for result in batch_results:
        if isinstance(result, Exception):
            print(f"⚠️ 警告: 规则批处理请求失败，忽略此结果。错误: {result}")
        else:
            rule_match_results.update(result)

    # 确保所有规则都有结果
    final_results: Dict[str, str] = {}
    for rule in rules:
        # LLM 返回的解释可能不包含 Rule X:，所以我们保留原始格式作为安全回退
        if rule in rule_match_results:
            final_results[rule] = rule_match_results[rule]
        else:
            final_results[rule] = f"No explanation available for Rule: {rule}"

    return final_results


# 5. 保存有效规则 (保持不变)
def save_matched_rules(effective_rules: Dict[str, List[Tuple[str, str]]], output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    for filename, rules_data in effective_rules.items():
        sub_folder = os.path.join(output_folder, "initial")
        os.makedirs(sub_folder, exist_ok=True)

        # 修正文件名，确保后缀正确
        base_name = os.path.splitext(filename)[0].replace('_rules_llm', '')  # 移除规则文件后缀以便于命名
        output_path = os.path.join(sub_folder, f"{base_name}_evaluated.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            for rule, explanation_string in rules_data:
                # 只有当解释不是默认值时才认为是“有效”的并写入
                if explanation_string and not explanation_string.startswith("No explanation available"):
                    f.write(f"Rule: {rule}\n")
                    f.write(f"Contradiction Explanation: {explanation_string}\n\n")

    print(f"\n✅ Successfully saved effective rules with contradiction explanations in: {output_folder}")


# -------------------- II. 主程序重构为异步 --------------------

async def main():
    # === ⏱️ 计时开始 ===
    overall_start = time.perf_counter()

    contradictions_file = 'dense_text.json'
    llm_folder = './dual_star/llm_rules'
    output_folder = './dual_star/llm_evaluations_v3'

    print(f"--- 🚀 评估流程开始 (批量并发模式) ---")
    print(f"--- ⚙️ 配置: 矛盾批次={CONTRADICTION_BATCH_SIZE}, 规则批次={RULES_BATCH_SIZE} ---")

    # 1. 加载矛盾数据
    try:
        contradictions = load_contradictions(contradictions_file)
        print(f"📦 已加载 {len(contradictions)} 条矛盾数据。")
    except Exception as e:
        print(f"🔴 错误: 加载矛盾文件失败: {e}")
        return

    # 2. 批量提取矛盾位置 (测量此阶段耗时)
    stage1_start = time.perf_counter()
    positions_list = await extract_contradiction_positions_batched(contradictions, llm)
    stage1_end = time.perf_counter()

    # 合并结果
    for i, c in enumerate(contradictions):
        c['contradiction_position'] = positions_list[i]

    # 3. 加载规则文件
    rules_by_file = load_rules_from_llm(llm_folder)
    total_rules_count = sum(len(r) for r in rules_by_file.values())

    if not rules_by_file:
        print("🔴 无法继续：没有找到任何规则文件。")
        return

    # 4. 评估规则阶段
    stage2_start = time.perf_counter()
    effective_rules: Dict[str, List[Tuple[str, str]]] = {}

    for filename, rules in rules_by_file.items():
        print(f"\n🔍 正在评估: {filename} ({len(rules)} 条规则)...")
        matching_results = await match_rules_to_contradictions_concurrent(rules, contradictions, llm)

        file_rules_data = []
        for rule in rules:
            explanation_string = matching_results.get(rule, "")
            file_rules_data.append((rule, explanation_string))
        effective_rules[filename] = file_rules_data

    stage2_end = time.perf_counter()

    # 5. 保存结果
    save_matched_rules(effective_rules, output_folder)

    # === 📊 生成性能统计报告 ===
    overall_end = time.perf_counter()

    total_duration = overall_end - overall_start
    s1_duration = stage1_end - stage1_start
    s2_duration = stage2_end - stage2_start

    print("\n" + "=" * 50)
    print("           ✨ 批量评估任务执行报告 ✨")
    print("=" * 50)
    print(f"📊 数据规模:")
    print(f"   - 矛盾条数: {len(contradictions)}")
    print(f"   - 规则总数: {total_rules_count}")
    print(f"   - 处理文件: {len(rules_by_file)} 个")
    print("-" * 50)
    print(f"⏱️ 耗时明细:")
    print(f"   - 阶段 1 (矛盾位置分类): {s1_duration:.2f} 秒")
    print(f"   - 阶段 2 (规则逻辑匹配): {s2_duration:.2f} 秒")
    print(f"   - 🚀 总执行耗时: {total_duration:.2f} 秒 (约 {total_duration / 60:.2f} 分钟)")
    print("-" * 50)
    print(f"📈 吞吐量指标:")
    if total_duration > 0:
        print(f"   - 规则处理速度: {(total_rules_count / total_duration):.2f} 条/秒")
        print(f"   - 每分钟处理量: {((total_rules_count + len(contradictions)) / total_duration * 60):.2f} 任务项/分钟")
    print("=" * 50)
    print(f"📅 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=" * 50)


if __name__ == '__main__':
    asyncio.run(main())