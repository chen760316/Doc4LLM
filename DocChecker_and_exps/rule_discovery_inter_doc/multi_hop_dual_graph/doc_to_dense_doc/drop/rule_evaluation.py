"""
从llm生成的规则中找到能检测矛盾的有效规则，保存在llm_evaluations文件夹下
"""
import json
import os
import re
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List

# 设定你的目录
OUTPUT_DIR = "../dual_star/llm"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


# 1. 读取 dense_text_cross.txt 文件，存储矛盾类型和句子对
def load_contradictions(file_path):
    contradictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        contradictions = json.load(f)
    return contradictions


# 2. 读取 llm 文件夹中的规则文件
def load_rules_from_llm(folder_path):
    rules_by_file = {}
    rule_start_pattern = re.compile(r'Rules:\s*')

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()

                # 查找 Rules: 后的部分
                match = rule_start_pattern.search(content)
                if match:
                    # 从 Rules: 后面开始提取规则
                    rules_section = content[match.end():].strip()

                    # 假设规则是按行分隔的，我们将其拆分成列表
                    rules_list = rules_section.split('\n')

                    # 去掉空行并添加到 rules 列表中
                    rules_by_file[filename] = [rule.strip() for rule in rules_list if rule.strip()]
    return rules_by_file


# 3. 构建评估规则的提示
def build_evaluation_prompt(rules_by_file, contradictions):
    prompt = "You are an expert in contradiction detection. Your task is to evaluate the following set of rules.\n"

    for filename, rules in rules_by_file.items():
        prompt += f"\n--- Rules from file: {filename} ---\n"
        for rule in rules:
            prompt += f"{rule}\n"

    prompt += "\n### Task:\nPlease evaluate the effectiveness of the above rules. For each rule, indicate whether it is effective or ineffective based on the provided contradictions. A rule is considered effective if it is related to any of the contradiction types in the given contradictions."

    # You can optionally provide the contradiction data here for better context.
    prompt += f"\n\nContradictions:\n{json.dumps(contradictions, ensure_ascii=False)}"

    return prompt


# 4. 使用 LLM 评估规则
def evaluate_rules_with_llm(rules_by_file, contradictions, llm):
    prompt = build_evaluation_prompt(rules_by_file, contradictions)

    system_message = (
        "You are a contradiction detection assistant. Your task is to evaluate the effectiveness "
        "of the provided logical rules based on the given contradictions. A rule is considered effective if it "
        "is related to any of the contradiction types in the provided contradictions."
    )

    # 提交给 LLM 进行评估
    response = llm.generate(
        user_message=prompt,
        system_message=system_message
    )

    # 解析 LLM 返回的结果
    evaluation_results = response.strip().split("\n")
    return evaluation_results


# 5. 解析 LLM 返回的评估结果
def parse_evaluation_results(evaluation_results, rules_by_file):
    effective_rules = {}
    ineffective_rules = {}

    # 解析每个文件的评估结果
    for idx, (filename, rules) in enumerate(rules_by_file.items()):
        effective_rules[filename] = []
        ineffective_rules[filename] = []

        for rule, result in zip(rules, evaluation_results[idx:]):
            if '→' not in rule:
                continue  # 跳过不完整的规则
            if 'effective' in result.lower():
                effective_rules[filename].append(rule)
            else:
                ineffective_rules[filename].append(rule)

    return effective_rules, ineffective_rules


# 6. 保存评估结果到文件（只保存有效的规则）
def save_evaluation_results(effective_rules, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename, rules in effective_rules.items():
        with open(os.path.join(output_folder, f"{filename}_effective.txt"), "w", encoding="utf-8") as f:
            for rule in rules:
                f.write(rule + "\n")

    print(f"\n✅ Successfully saved effective rules in: {output_folder}")


# 主程序入口
def main():
    contradictions_file = '../dense_text_cross.txt'  # 替换为实际文件路径
    llm_folder = './dual_star/llm'  # 替换为实际文件夹路径
    output_folder = './dual_star/llm_evaluations'  # 输出评估结果的文件夹路径

    # 加载矛盾数据
    contradictions = load_contradictions(contradictions_file)

    # 加载 llm 文件夹中的所有规则
    rules_by_file = load_rules_from_llm(llm_folder)

    # 使用 LLM 批量评估规则
    evaluation_results = evaluate_rules_with_llm(rules_by_file, contradictions, llm)

    # 解析评估结果
    effective_rules, ineffective_rules = parse_evaluation_results(evaluation_results, rules_by_file)

    # 保存评估结果
    save_evaluation_results(effective_rules, output_folder)


if __name__ == '__main__':
    main()
