import os
import json
import re
import asyncio
from dotenv import load_dotenv


# 假设 OpenAIClient 是一个可以正常工作的类，用于消除 API 警告
class OpenAIClient:
    def __init__(self, model, temperature, top_p, max_tokens):
        # 简化初始化，避免实际连接
        pass

    def generate(self, user_message, system_message):
        # 这是一个占位符，因为在这个脚本中 LLM 实际上没有被调用
        return ""


from typing import List, Dict, Any, Tuple

# 设定目录
INPUT_DIR = "dual_star/llm_evaluations_v3/effective"
OUTPUT_DIR = "dual_star/llm_evaluations_v3/effective_with_center_name"  # 修改输出目录名以区分
JSON_INPUT_DIR = "dual_star_json_with_attrs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化环境
load_dotenv()
# 消除 API 警告，确保环境变量已设置
if os.getenv("OPENAI_API_KEY") is None:
    print("⚠️ 警告: OPENAI_API_KEY 未设置。LLMClient 初始化将失败，但文件处理可继续。")

llm = OpenAIClient(
    model="gpt-4o-mini",
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


# -------------------- II. 核心辅助函数 (同步) --------------------

def load_entity_names(json_file_path: str) -> Tuple[str, str]:
    """
    【修正点】从 JSON 文件中提取 x0 (第一个 center) 和 y0 (第二个 center) 的实体名称。
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        stars = data.get("stars", [])
        if len(stars) < 2:
            # 如果 star 数量不足 2，则无法提取两个 center
            return "x0_default", "y0_default"

        star_one = stars[0]
        star_two = stars[1]  # 获取第二个 star

        # x0 提取第一个 center
        x0 = star_one.get("center")
        # y0 提取第二个 center
        y0 = star_two.get("center")

        # 确保 x0 和 y0 不为空
        x0 = x0 if x0 else "x0_default"
        y0 = y0 if y0 else "y0_default"

        return x0, y0
    except Exception as e:
        print(f"⚠️ 警告: 读取实体名称文件 {os.path.basename(json_file_path)} 失败: {e}")
        return "x0_default", "y0_default"


def read_rules_from_files(input_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    读取指定文件夹下的 JSON 规则文件，并返回规则列表和解释。
    """
    rules_by_file = {}

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                    if not data or not isinstance(data, list) or not data[0]:
                        continue

                    reflecting_rules = data[0].get('reflecting_rules', [])

                    rules_data = []
                    for rule_line in reflecting_rules:
                        cleaned_rule = re.sub(r'^\s*-\s*\d+\.\s*', '', rule_line).strip()
                        rules_data.append((cleaned_rule, ""))

                    rules_by_file[file_name] = rules_data

            except Exception as e:
                print(f"⚠️ 警告: 读取或解析文件 {file_name} 时出错: {e}")

    return rules_by_file


def substitute_entities_in_rules(rules_data: List[Tuple[str, str]], x0: str, y0: str) -> List[Tuple[str, str]]:
    """
    替换规则数据中的 Q[x0, y0] 占位符。
    """

    def replace_q_scope(rule: str, x0_name: str, y0_name: str) -> str:
        x0_sub = f'"{x0_name}"'
        y0_sub = f'"{y0_name}"'

        # 替换 Q[x0, y0]
        return rule.replace('Q[x0, y0]', f'Q[{x0_sub}, {y0_sub}]')

    new_rules_data = []
    for rule, explanation in rules_data:
        new_rule = replace_q_scope(rule, x0, y0)
        new_rules_data.append((new_rule, explanation))

    return new_rules_data


# -------------------- III. 异步主处理函数 (重写为同步) --------------------

def process_single_file_entity_substitution(file_name: str, input_dir: str, output_dir: str, rules_by_file: Dict[str, List[Tuple[str, str]]]):
    """
    处理单个文件的实体替换逻辑并保存结果（同步）。
    """

    # 【已修正】路径构造逻辑：确保正确映射到 dual_star_json_with_attrs 目录下的文件
    base_name = re.sub(r'_evaluated_refined\.json$', '', file_name)

    json_file_name = f"{base_name}.json"
    json_file_path = os.path.join(JSON_INPUT_DIR, json_file_name)

    print(f"🔗 [INFO] Mapping {file_name} to entity file: {json_file_path}")

    # 2. 提取实体名称 (x0=center 1, y0=center 2)
    x0, y0 = load_entity_names(json_file_path)

    # 检查是否使用了默认值（例如 stars 数量不足 2）
    if x0 == "x0_default" or y0 == "y0_default":
        print(f"⚠️ [SKIP] {file_name}: 无法从 {json_file_name} 提取两个核心实体 (center)。")
        return

    # 3. 获取规则数据
    rules_data = rules_by_file.get(file_name, [])

    if not rules_data:
        print(f"⚠️ [SKIP] {file_name}: 没有找到规则数据。")
        return

    # 4. 替换实体 (同步)
    substituted_rules_data = substitute_entities_in_rules(rules_data, x0, y0)

    # 5. 保存数据 (同步)
    output_file_path = os.path.join(output_dir, f"{base_name}_final_rules.txt")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"### Entity Mapping ###\n")
            f.write(f"x0: \"{x0}\"\n")
            f.write(f"y0: \"{y0}\"\n\n")
            f.write("### Rules with Substituted Entities ###\n")

            for rule, explanation in substituted_rules_data:
                f.write(f"Rule: {rule}\n")
                f.write("\n")

        print(f"✅ [SUCCESS] Substituted rules for {file_name} saved to {os.path.basename(output_file_path)}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to save data for {file_name}: {e}")


# -------------------- IV. 异步主函数 --------------------

async def main_concurrent():
    print("--- 规则实体替换流程开始 (并发处理 I/O，但核心逻辑为同步) ---")

    # 1. 读取规则文件 (同步操作)
    rules_by_file = read_rules_from_files(INPUT_DIR)

    if not rules_by_file:
        print("🔴 无法继续：在输入目录中未找到任何规则文件。")
        return

    # 2. 准备并发任务列表
    tasks = []

    for file_name in rules_by_file.keys():
        task = asyncio.to_thread(
            process_single_file_entity_substitution,
            file_name=file_name,
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            rules_by_file=rules_by_file
        )
        tasks.append(task)

    if not tasks:
        print("⚠️ 警告: 没有找到匹配的输入文件来处理。")
        return

    print(f"--- 🚀 发现 {len(tasks)} 个文件需要处理，启动并发文件处理 ---")

    # 3. 并发执行所有任务
    await asyncio.gather(*tasks)

    print("\n--- ✅ 所有文件处理完毕 ---")


if __name__ == "__main__":
    asyncio.run(main_concurrent())