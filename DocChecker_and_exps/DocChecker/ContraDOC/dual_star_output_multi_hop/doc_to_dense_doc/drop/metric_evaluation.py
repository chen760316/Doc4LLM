import json
import re
import os
import asyncio
from glob import glob
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Tuple, Any, Set
import threading
from functools import partial
import sys
from contextlib import redirect_stdout  # 引入重定向工具

# --- 配置和初始化 ---
# 设定规则目录
RULE_DIR = 'dual_star/llm_learning_rules'
CONTRADICTION_FILE = 'dense_text.txt'  # 假设这是您的矛盾数据文件路径
OUTPUT_FILE = 'result.txt'  # 设定输出文件名
# 新增评估规则目录
EVAL_RULE_DIR = 'dual_star/llm_evaluations_v3/effective_with_center_name'

# 初始化环境
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=3000
)

# 用于保护共享资源的锁
print_lock = threading.Lock()


# -------------------- 异步包装器 --------------------

async def _llm_generate_async(llm: OpenAIClient, user_message: str, system_message: str) -> str:
    """使用 asyncio.to_thread 包装同步的 LLM 调用。"""
    try:
        return await asyncio.to_thread(
            llm.generate,
            user_message=user_message,
            system_message=system_message
        )
    except Exception as e:
        with print_lock:
            print(f"❌ 警告: LLM 请求失败: {e}")
        return None


# -------------------- I. LLM 交互函数 (异步化) --------------------

async def get_contradiction_type_async(s1: str, s2: str) -> str:
    """异步获取矛盾类型。"""
    rules_prompt = f"""
    You are an expert in detecting contradiction types between two sentences.
    Sentence 1: {s1}
    Sentence 2: {s2}

    Please classify the contradiction type between these two sentences.
    Provide only the contradiction type, no explanations. 
    Possible contradiction types: 'Numeric', 'Temporal', 'Factual', 'Content', 'Causal', 'Negation', 'Relation'.
    """
    system_message = "You are a helpful assistant that analyzes contradictions between sentences."

    response = await _llm_generate_async(llm, user_message=rules_prompt, system_message=system_message)

    return response.strip() if response else "Unknown"


async def extract_entities_and_attributes_from_llm_async(s1: str, s2: str) -> Dict[str, List[str]]:
    """异步提取实体和属性。"""
    prompt = f"""
    Please analyze the following two sentences and extract the entities (e.g., persons, locations, objects, etc.) and their attributes (e.g., actions, relations, descriptors).

    Sentence 1: "{s1}"
    Sentence 2: "{s2}"

    For each sentence, please return the following:
    - Entities: [list of entities in sentence]
    - Attributes: [list of attributes associated with those entities]

    Output the results using JSON format only:
    {{
        "sentence_1": {{"entities": [list of entities], "attributes": [list of attributes]}},
        "sentence_2": {{"entities": [list of entities], "attributes": [list of attributes]}}
    }}
    """
    system_message = "You are an expert at analyzing sentences and extracting entities and attributes. Your output MUST be a valid JSON object."

    response = await _llm_generate_async(llm, user_message=prompt, system_message=system_message)

    if not response:
        return {}

    try:
        # 尝试解析 JSON
        data = json.loads(response)
        return {
            "entities_s1": data.get("sentence_1", {}).get("entities", []),
            "attributes_s1": data.get("sentence_1", {}).get("attributes", []),
            "entities_s2": data.get("sentence_2", {}).get("entities", []),
            "attributes_s2": data.get("sentence_2", {}).get("attributes", [])
        }
    except json.JSONDecodeError as e:
        # 这里的 print() 将被重定向捕获
        with print_lock:
            print(f"Error parsing JSON from entity extraction: {e}. Raw response: {response[:200]}...")
        # 如果解析失败，进行简单的文本解析回退（虽然 JSON 更可靠）
        return _extract_entities_and_attributes_fallback(response)


def _extract_entities_and_attributes_fallback(response: str) -> Dict[str, List[str]]:
    """文本解析回退逻辑 (与原代码保持一致)"""

    def parse_list(text):
        match = re.search(r'\[(.*?)\]', text)
        if match:
            # 简单地用逗号分割并清理
            return [item.strip().strip('"') for item in match.group(1).split(',')]
        return []

    try:
        data = {
            "entities_s1": parse_list(re.search(r'Sentence 1:\s*Entities:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE).group(1)) if re.search(r'Sentence 1:\s*Entities:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE) else [],
            "attributes_s1": parse_list(re.search(r'Sentence 1:.*?Attributes:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE).group(1)) if re.search(r'Sentence 1:.*?Attributes:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE) else [],
            "entities_s2": parse_list(re.search(r'Sentence 2:\s*Entities:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE).group(1)) if re.search(r'Sentence 2:\s*Entities:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE) else [],
            "attributes_s2": parse_list(re.search(r'Sentence 2:.*?Attributes:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE).group(1)) if re.search(r'Sentence 2:.*?Attributes:\s*\[(.*?)\]', response, re.DOTALL | re.IGNORECASE) else [],
        }
        return data
    except Exception:
        return {
            "entities_s1": [], "attributes_s1": [],
            "entities_s2": [], "attributes_s2": []
        }


# -------------------- II. 数据加载和处理 (同步) --------------------

def load_contradictions(file_path: str) -> List[Dict[str, str]]:
    """读取矛盾数据文件。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"❌ 错误: 矛盾文件未找到: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"❌ 错误: 矛盾文件 {file_path} 不是有效的 JSON 格式。")
        return []


def load_rules_from_dir(dir_path: str) -> List[str]:
    """
    遍历目录，从所有 *.txt 文件中提取规则。
    处理两种规则格式：
    1. RULE_DIR (如: dual_star/llm_learning_rules) 下的文件，匹配 'Rules:' 后的 Q[...]。
    2. EVAL_RULE_DIR (如: dual_star/llm_evaluations_v3/effective_with_center_name)
       下的文件，匹配 'Rule: Q[' 开头的行。
    """
    all_rules = []

    # --- 1. 加载 LLM Learning Rules (原始格式) ---
    rule_files_llm = glob(os.path.join(dir_path, '*_rules_llm.txt'))

    for file_path in rule_files_llm:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 匹配 Rules: 部分
            rules_section_match = re.search(r"Rules:\s*\n((?:Q\[.*?\]\(.*?\)\s*\n?)+)", content, re.DOTALL)

            if rules_section_match:
                rules_text = rules_section_match.group(1).strip()
                rules = [
                    line.strip()
                    for line in rules_text.splitlines()
                    if line.strip().startswith('Q[')
                ]
                all_rules.extend(rules)

        except Exception as e:
            print(f"❌ 读取或解析规则文件 {os.path.basename(file_path)} 失败: {e}")

    # --- 2. 加载 Evaluation Rules (新格式) ---
    # 使用递归 glob 查找所有子文件
    eval_rule_files = glob(os.path.join(EVAL_RULE_DIR, '**', '*.txt'), recursive=True)

    for file_path in eval_rule_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 逐行读取，以减少内存占用，并方便匹配
                for line in f:
                    line = line.strip()
                    # 精确匹配 'Rule: Q[...]'
                    if line.startswith('Rule: Q['):
                        # 移除 "Rule: " 前缀，只保留规则部分
                        rule = line[len('Rule: '):].strip()
                        all_rules.append(rule)
        except Exception as e:
            print(f"❌ 读取或解析评估规则文件 {os.path.basename(file_path)} 失败: {e}")

    # 去重
    unique_rules = list(set(all_rules))
    print(f"✅ 从 {len(rule_files_llm)} 个旧规则文件和 {len(eval_rule_files)} 个评估规则文件中加载了 {len(unique_rules)} 条唯一规则。")
    return unique_rules


# -------------------- III. 规则转换 (异步批处理) --------------------

async def convert_rules_to_natural_language_async(rules: List[str], batch_size: int = 20) -> List[str]:
    """异步将规则列表分批转换为自然语言。"""
    system_message = "You are a helpful assistant that rephrases formal rules into clear, natural language."
    tasks = []

    for i in range(0, len(rules), batch_size):
        batch = rules[i:i + batch_size]

        batch_prompt = "You are an expert at explaining formal rules clearly in natural language.\n\n"
        batch_prompt += "Please rewrite each of the following rules into clear, simple natural language:\n\n"

        for idx, rule in enumerate(batch, start=i + 1):
            batch_prompt += f"{idx}. {rule}\n"

        batch_prompt += "\nReturn the rewritten rules in the same numbered list format."

        # 创建异步任务
        task = _llm_generate_async(llm, user_message=batch_prompt, system_message=system_message)
        tasks.append(task)

    responses = await asyncio.gather(*tasks)

    natural_language_rules = []
    for response in responses:
        if response:
            # 提取出每条规则文本
            lines = [
                re.sub(r'^\s*\d+\.\s*', '', line).strip()
                for line in response.strip().splitlines()
                if line.strip() and re.match(r'^\s*\d+\.\s*', line)
            ]
            natural_language_rules.extend(lines)

    # 这里的 print() 将被重定向捕获
    with print_lock:
        print(f"\n✅ 成功转换 {len(natural_language_rules)} 条规则为自然语言。")
        # 打印部分转换后的规则示例
        for rule in natural_language_rules[:5]:
            print(f"Example: {rule}")

    return natural_language_rules


# -------------------- IV. 匹配和覆盖率评估 --------------------

def is_rule_covering_contradiction(rule: str, entities_s1: List[str], entities_s2: List[str], attributes_s1: List[str], attributes_s2: List[str], threshold: float = 0.5) -> bool:
    """
    基于阈值判断规则是否覆盖了矛盾。
    """
    # 修正：在调用 .lower() 之前，确保元素是字符串 (str)
    all_entities = [
        str(e).lower()
        for e in entities_s1 + entities_s2
        if e and isinstance(e, (str, dict))
    ]
    all_attributes = [
        str(a).lower()
        for a in attributes_s1 + attributes_s2
        if a and isinstance(a, (str, dict))
    ]

    total_items = len(all_entities) + len(all_attributes)
    if total_items == 0:
        return False

    rule_lower = rule.lower()
    matched_count = 0

    for entity in all_entities:
        if entity in rule_lower:
            matched_count += 1

    for attribute in all_attributes:
        if attribute in rule_lower:
            matched_count += 1

    coverage_ratio = matched_count / total_items

    return coverage_ratio >= threshold


async def check_rule_coverage_async(contradictions: List[Dict[str, str]], rules: List[str]):
    """异步执行规则覆盖率检查。"""

    # 1. 异步转换所有规则为自然语言
    natural_rules = await convert_rules_to_natural_language_async(rules)

    if not natural_rules:
        print("❌ 规则转换失败，无法继续评估。")
        return

    # 2. 并发处理所有矛盾句子对
    coverage_tasks = []
    # 使用 original_rules 的索引来保证 natural_rules 和 original_rules 的对应关系
    # 因为 natural_rules 是由原始 rules 异步生成的，其顺序应该保持一致
    original_rules = rules

    for contradiction in contradictions:
        s1 = contradiction['sentence_1']
        s2 = contradiction['sentence_2']
        # 创建单个矛盾的分析任务
        task = analyze_single_contradiction_async(s1, s2, natural_rules, original_rules)
        coverage_tasks.append(task)

    results = await asyncio.gather(*coverage_tasks)

    # 3. 汇总结果
    covered_contradictions = []
    uncovered_contradictions = []
    covered_types = set()
    uncovered_types = set()

    for result in results:
        if not result: continue

        s1, s2, contradiction_type, rule_covered, natural_rule, original_rule = result

        if rule_covered:
            covered_types.add(contradiction_type)
            covered_contradictions.append({
                'contradiction': f"Sentence 1: {s1} / Sentence 2: {s2}",
                'type': contradiction_type,
                'rule': natural_rule,
                'original_rule': original_rule
            })
        else:
            uncovered_types.add(contradiction_type)
            uncovered_contradictions.append({
                'contradiction': f"Sentence 1: {s1} / Sentence 2: {s2}",
                'type': contradiction_type
            })

    total_count = len(contradictions)
    covered_count = len(covered_contradictions)

    # 4. 输出总结
    print("\n" + "=" * 30 + " 覆盖率评估总结 " + "=" * 30)
    print(f"总矛盾对数量: {total_count}")
    print(f"被规则覆盖的矛盾对数量: {covered_count}")
    print(f"覆盖率: {(covered_count / total_count) * 100:.2f}%")
    print("-" * 75)
    print(f"覆盖的矛盾类型: {', '.join(covered_types)}")
    print(f"未覆盖的矛盾类型: {', '.join(uncovered_types - covered_types)}")
    print("=" * 75)

    # 输出具体覆盖的矛盾和规则 (仅展示前5个)
    print("\n✅ 部分被覆盖的矛盾和对应规则 (前 5 个):")
    for i, contradiction in enumerate(covered_contradictions[:5]):
        print(f"--- {i + 1}. 矛盾类型: {contradiction['type']} ---")
        print(f"句子对: {contradiction['contradiction']}")
        print(f"自然语言规则: {contradiction['rule']}")
        print(f"原始规则: {contradiction['original_rule']}")

    # 输出未覆盖的矛盾 (仅展示前5个)
    print("\n❌ 部分未被覆盖的矛盾 (前 5 个):")
    for i, contradiction in enumerate(uncovered_contradictions[:5]):
        print(f"--- {i + 1}. 矛盾类型: {contradiction['type']} ---")
        print(f"句子对: {contradiction['contradiction']}")


async def analyze_single_contradiction_async(s1: str, s2: str, natural_rules: List[str], original_rules: List[str]) -> Tuple[str, str, str, bool, str, str]:
    """分析单个矛盾，并检查其是否被规则覆盖。"""

    # 1. 并发获取矛盾类型、实体和属性
    type_task = get_contradiction_type_async(s1, s2)
    entity_task = extract_entities_and_attributes_from_llm_async(s1, s2)

    contradiction_type, entities_and_attributes = await asyncio.gather(type_task, entity_task)

    if not entities_and_attributes:
        return (s1, s2, contradiction_type, False, "", "")  # 提取失败，视为未覆盖

    entities_s1 = entities_and_attributes.get("entities_s1", [])
    attributes_s1 = entities_and_attributes.get("attributes_s1", [])
    entities_s2 = entities_and_attributes.get("entities_s2", [])
    attributes_s2 = entities_and_attributes.get("attributes_s2", [])

    # 2. 检查规则覆盖
    rule_covered = False
    natural_rule_match = ""
    original_rule_match = ""

    for idx, natural_rule in enumerate(natural_rules):
        # 确保索引在 original_rules 范围内，以防万一
        if idx >= len(original_rules):
            continue

        if is_rule_covering_contradiction(natural_rule, entities_s1, entities_s2, attributes_s1, attributes_s2, threshold=0.5):
            rule_covered = True
            natural_rule_match = natural_rule
            original_rule_match = original_rules[idx]
            break  # 找到匹配规则即停止

    return (s1, s2, contradiction_type, rule_covered, natural_rule_match, original_rule_match)


# -------------------- V. 主执行函数 --------------------

async def analyze_coverage_main(contradiction_file: str, rule_dir: str):
    """主执行函数：加载数据并启动异步评估。"""

    # 1. 加载所有数据 (同步操作)
    # 这里的 load_rules_from_dir 已经修改为加载两个目录下的规则
    contradictions = load_contradictions(contradiction_file)
    rules = load_rules_from_dir(rule_dir)

    if not contradictions:
        print("评估终止：未加载到任何矛盾数据。")
        return
    if not rules:
        print("评估终止：未加载到任何规则数据。")
        return

    # 2. 执行异步覆盖率检查
    await check_rule_coverage_async(contradictions, rules)


if __name__ == "__main__":
    # 使用 contextlib.redirect_stdout 将所有 print 输出重定向到文件
    try:
        # 确保 EVAL_RULE_DIR 在当前工作目录下存在，或者使用绝对路径
        if not os.path.exists(EVAL_RULE_DIR):
            print(f"⚠️ 警告: 规则目录 {EVAL_RULE_DIR} 不存在，请检查路径。仅加载 {RULE_DIR} 下的规则。", file=sys.stderr)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print(f"--- 规则覆盖率评估开始 ({os.path.basename(CONTRADICTION_FILE)} vs {RULE_DIR} & {EVAL_RULE_DIR}) ---")
                asyncio.run(analyze_coverage_main(CONTRADICTION_FILE, RULE_DIR))
                print("--- 规则覆盖率评估结束 ---")
    except Exception as e:
        # 如果程序在重定向之外崩溃，这里的 print() 才会显示在控制台
        print(f"程序执行出错，无法写入文件: {e}", file=sys.stderr)