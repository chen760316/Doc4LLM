import json
import os
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
# 假设知识图谱生成器中的 OpenAIClient 已经定义
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple, Set as TypeSet

# 设定目录
OUTPUT_DIR = "dual_star/llm_evaluations_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化环境 (保持不变)
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 假设 OpenAIClient 已正确初始化
llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# 定义并发任务的最大数量 (可以根据你的API限制和带宽调整)
MAX_CONCURRENT_LLM_CALLS = 10


# -------------------- I. LLM 核心函数重构 --------------------

# 1. 读取矛盾数据 (保持不变)
def load_contradictions(file_path: str) -> List[Dict[str, Any]]:
    # ... (保持用户提供的 load_contradictions 函数体)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 2. 异步提取单个矛盾的位置
# 由于 llm.generate 是同步的，我们用 to_thread 包装它
async def _extract_single_position(reason: str, llm: OpenAIClient) -> str:
    """异步包装器：为单个原因生成 prompt 并调用 LLM。"""
    prompt = f"Contradiction: {reason}\n"

    # 使用 asyncio.to_thread 来在单独的线程中运行同步的 llm.generate 调用
    # 这样主事件循环就不会被阻塞
    response = await asyncio.to_thread(
        llm.generate,
        user_message=prompt,
        system_message="You are an expert in contradiction analysis. Your task is to extract the specific location or type of contradiction from the provided reasons. Output ONLY the extracted type, one per line."
    )

    # 清理响应，只保留第一个类型
    positions = [pos.strip() for pos in response.strip().split('\n') if pos.strip()]
    return positions[0] if positions else "Missing_LLM_Response"


# 2. 并发提取矛盾位置的主函数
async def extract_contradiction_positions_concurrent(contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> List[Dict[str, Any]]:
    print(f"🧠 正在并发请求 {len(contradictions)} 个矛盾位置...")

    tasks = []
    for contradiction in contradictions:
        reason = contradiction.get('reason', 'No reason provided')
        # 创建一个异步任务列表
        tasks.append(_extract_single_position(reason, llm))

    # 使用 asyncio.gather 并行执行所有任务
    # limit=MAX_CONCURRENT_LLM_CALLS 可以通过 Semaphore 实现，这里使用简单的 gather
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 将结果合并回矛盾列表
    for i, contradiction in enumerate(contradictions):
        result = results[i]
        if isinstance(result, Exception):
            print(f"⚠️ 警告: LLM 请求失败，使用默认值。错误: {result}")
            contradiction['contradiction_position'] = "LLM_Request_Failed"
        else:
            contradiction['contradiction_position'] = result

    return contradictions


# 3. 加载规则文件并与矛盾进行匹配 (函数名修正)
def load_rules_from_llm(folder_path: str) -> Dict[str, List[str]]:
    # ... (保持用户提供的 load_rules_from_llm 函数体)
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


# 4. 异步匹配单个规则到矛盾类型
async def _match_single_rule(rule: str, simplified_contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> Tuple[str, str]:
    """异步包装器：为单个规则生成 prompt 并调用 LLM。"""
    prompt = f"Rule 1: {rule}\nAll Contradictions (Simplified): {json.dumps(simplified_contradictions, ensure_ascii=False)}\n"

    response = await asyncio.to_thread(
        llm.generate,
        user_message=prompt,
        system_message="You are an expert in contradiction detection. Your task is to evaluate if the rule matches any contradiction type based on the provided contradictions. Explain the reasoning in full sentences.\nFor the rule, return the explanation in the following format:\nRule 1: <Explanation for Rule 1>\nIf a rule has no explanation, use 'No explanation available for Rule 1.'"
    )

    # 尝试解析解释字符串
    match = re.search(r'Rule 1:\s*(.*)', response, re.DOTALL)
    explanation = match.group(1).strip() if match else response.strip()

    return rule, explanation


# 4. 并发匹配规则的主函数
async def match_rules_to_contradictions_concurrent(rules: List[str], contradictions: List[Dict[str, Any]], llm: OpenAIClient) -> Dict[str, str]:
    # 简化矛盾数据 (只在内存中创建一次)
    simplified_contradictions = [{"id": i, "type": c.get('contradiction_type'), "position": c.get('contradiction_position')} for i, c in enumerate(contradictions)]

    print(f"🧠 正在并发请求 {len(rules)} 条规则的解释...")

    tasks = []
    for rule in rules:
        # 创建一个异步任务列表
        tasks.append(_match_single_rule(rule, simplified_contradictions, llm))

    # 使用 asyncio.gather 并行执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rule_match_results: Dict[str, str] = {}

    for result in results:
        if isinstance(result, Exception):
            # 处理失败的请求
            # 由于我们不知道是哪条规则失败了，这里只能记录错误，但无法精确关联
            print(f"⚠️ 警告: 规则匹配请求失败，忽略此结果。错误: {result}")
        else:
            rule, explanation = result
            rule_match_results[rule] = explanation

    # 确保没有匹配到的规则返回默认值
    for rule in rules:
        if rule not in rule_match_results:
            rule_match_results[rule] = f"No explanation available for Rule: {rule}"

    return rule_match_results


# 5. 保存有效规则 (保持不变，注意它现在期望 explanation_string)
def save_matched_rules(effective_rules: Dict[str, List[Tuple[str, str]]], output_folder: str):
    os.makedirs(output_folder, exist_ok=True)

    for filename, rules_data in effective_rules.items():
        sub_folder = os.path.join(output_folder, "initial")
        os.makedirs(sub_folder, exist_ok=True)

        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(sub_folder, f"{base_name}_effective_with_contradictions.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            for rule, explanation_string in rules_data:
                # explanation_string 作为 Contradiction Explanation 写入
                if explanation_string and explanation_string != f"No explanation available for Rule: {rule}":
                    f.write(f"Rule: {rule}\n")
                    f.write(f"Contradiction Explanation: {explanation_string}\n\n")

    print(f"\n✅ Successfully saved effective rules with contradiction explanations in: {output_folder}")


# -------------------- II. 主程序重构为异步 --------------------

async def main():
    contradictions_file = 'dense_text.txt'
    llm_folder = './dual_star/llm_rules'
    output_folder = './dual_star/llm_evaluations_v3'

    print(f"--- 评估流程开始 (并发模式) ---")

    # 1. 加载矛盾数据 (同步操作)
    try:
        contradictions = load_contradictions(contradictions_file)
        print(f"Loaded {len(contradictions)} contradictions.")
    except Exception as e:
        print(f"🔴 错误: 加载矛盾文件失败: {e}")
        return

    # 2. 通过 LLM 并发提取矛盾位置
    contradictions_with_positions = await extract_contradiction_positions_concurrent(contradictions, llm)

    # 3. 加载规则文件 (同步操作)
    rules_by_file = load_rules_from_llm(llm_folder)

    if not rules_by_file:
        print("🔴 无法继续：没有找到任何规则文件。")
        return

    effective_rules: Dict[str, List[Tuple[str, str]]] = {}

    for filename, rules in rules_by_file.items():
        print(f"\n--- 评估文件: {filename} ({len(rules)} 条规则) ---")

        # 4. 使用 LLM 并发判断规则与矛盾的匹配
        matching_results = await match_rules_to_contradictions_concurrent(rules, contradictions_with_positions, llm)

        # 5. 直接保存规则和匹配的解释字符串
        file_rules_data = []
        for rule in rules:
            explanation_string = matching_results.get(rule, f"No explanation available for Rule: {rule}")
            file_rules_data.append((rule, explanation_string))

        effective_rules[filename] = file_rules_data

    # 5. 保存有效规则 (同步操作)
    save_matched_rules(effective_rules, output_folder)
    print("--- 评估流程完成 ---")


if __name__ == '__main__':
    # 使用 asyncio.run() 来运行顶层的异步 main 函数
    asyncio.run(main())