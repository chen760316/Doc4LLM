import json
import os
import re
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List

# 设定目录
OUTPUT_DIR = "dual_star/llm_evaluations_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化环境
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


# 1. 读取矛盾数据
def load_contradictions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 2. 通过 LLM 从 reason 字段中提取矛盾发生的位置
def extract_contradiction_positions(contradictions, llm):
    prompt = "For each contradiction reason below, please identify the location or specific aspect of the contradiction (e.g., 'Numeric', 'Temporal', 'Emotional', etc.).\n"

    for contradiction in contradictions:
        prompt += f"Contradiction: {contradiction['reason']}\n"

    # 提交给 LLM 提取矛盾位置
    response = llm.generate(
        user_message=prompt,
        system_message="You are an expert in contradiction analysis. Your task is to extract the specific location or type of contradiction from the provided reasons."
    )

    # 返回的响应中提取矛盾位置
    positions = response.strip().split('\n')

    for i, contradiction in enumerate(contradictions):
        contradiction['contradiction_position'] = positions[i].strip()  # 将提取的矛盾位置添加到数据中

    return contradictions


# 3. 加载规则文件并与矛盾进行匹配
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

                    # 假设规则是按行分隔的
                    rules_list = rules_section.split('\n')

                    # 去掉空行并添加到规则列表
                    rules_by_file[filename] = [rule.strip() for rule in rules_list if rule.strip()]
    return rules_by_file


import json
import re


def match_rules_to_contradictions(rules, contradictions, llm):
    # 创建一个存储所有规则及其匹配结果的字典
    rule_match_results = {}

    prompt = "Please evaluate the following rules against the provided contradictions. For each rule, determine if it matches any contradiction and explain why in full sentences.\n"

    # 为每条规则提供详细的矛盾数据，并要求 LLM 给出完整的解释
    for i, rule in enumerate(rules):
        prompt += f"Rule {i + 1}: {rule}\nContradictions: {json.dumps(contradictions, ensure_ascii=False)}\n"

    # 提交给 LLM 进行匹配，并要求返回完整的规则和解释，强制格式
    response = llm.generate(
        user_message=prompt,
        system_message="You are an expert in contradiction detection. Your task is to evaluate if each rule matches any contradiction type based on the provided contradictions. Please explain the reasoning in full sentences.\nFor each rule, return the explanation in the following format:\nRule 1: <Explanation for Rule 1>\nRule 2: <Explanation for Rule 2>\nIf a rule has no explanation, use 'No explanation available for Rule X.'"
    )

    # 解析返回的 response，按换行符分割
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

    # 按规则顺序提取每条规则和解释
    for i, rule in enumerate(rules):
        if i < len(lines):
            explanation = lines[i].strip()
        else:
            # 如果没有解释，则返回默认的解释
            explanation = f"No explanation available for Rule {i + 1}."

        # 将规则与对应的解释存入字典
        rule_match_results[rule] = explanation

    return rule_match_results


# 5. 保存有效规则
def save_matched_rules(effective_rules, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename, rules in effective_rules.items():
        with open(os.path.join(output_folder, f"{filename}_effective_with_contradictions.txt"), "w", encoding="utf-8") as f:
            for rule, contradiction_types in rules:
                # 只有在矛盾类型不为空时才写入文件
                if contradiction_types:
                    f.write(f"Rule: {rule}\n")
                    # 将逗号替换为空格
                    f.write(f"Contradiction Types: {' '.join(contradiction_types)}\n\n")

    print(f"\n✅ Successfully saved effective rules with contradiction types in: {output_folder}")




# 主程序入口
def main():
    contradictions_file = 'dense_text_cross.txt'  # 替换为实际文件路径
    llm_folder = './dual_star/llm'  # 替换为实际文件夹路径
    output_folder = './dual_star/llm_evaluations_v3/initial'  # 输出评估结果的文件夹路径

    # 加载矛盾数据
    contradictions = load_contradictions(contradictions_file)

    # 通过 LLM 提取矛盾位置
    contradictions_with_positions = extract_contradiction_positions(contradictions, llm)

    # 加载规则文件
    rules_by_file = load_rules_from_llm(llm_folder)

    # 存储有效规则及其对应的矛盾类型
    effective_rules = {}

    for filename, rules in rules_by_file.items():
        effective_rules[filename] = []

        # 使用 LLM 判断规则与矛盾的匹配
        matching_results = match_rules_to_contradictions(rules, contradictions_with_positions, llm)

        # 直接保存规则和匹配的矛盾类型
        for rule in rules:

            matched_contradictions = matching_results.get(rule, [])

            # 直接保存规则和对应的矛盾类型，不做任何条件判断
            effective_rules[filename].append((rule, matched_contradictions))

    # 保存有效规则
    save_matched_rules(effective_rules, output_folder)




if __name__ == '__main__':
    main()
