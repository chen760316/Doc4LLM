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


# 3. 将规则转换为自然语言（一次性转换所有规则）
def convert_rules_to_natural_language(rules, llm):
    prompt = "Please rephrase the following rules into natural language:\n" + "\n".join(rules)

    # 提交给 LLM 进行转化
    response = llm.generate(
        user_message=prompt,
        system_message="You are a language expert. Your task is to rephrase the following logical rules into clear natural language statements."
    )

    return response.strip().split('\n')


# 4. 使用 LLM 判断规则是否与矛盾类型匹配
def rule_matches_contradictions_with_llm(rules_natural_language, contradictions, llm):
    prompt = "For each of the following natural language rules, please determine whether it matches any of the contradiction types and provide the corresponding contradiction type.\n\n"

    for rule in rules_natural_language:
        prompt += f"Rule: {rule}\nContradictions: {json.dumps(contradictions, ensure_ascii=False)}\n\n"

    # 提交给 LLM 进行匹配
    response = llm.generate(
        user_message=prompt,
        system_message="You are an expert in contradiction detection. Your task is to evaluate if each rule matches any contradiction types based on the provided contradictions."
    )

    return response.strip().split('\n')


# 5. 保存有效规则及其对应的矛盾类型
def save_matched_rules(effective_rules, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename, rules in effective_rules.items():
        with open(os.path.join(output_folder, f"{filename}_effective_with_contradictions.txt"), "w", encoding="utf-8") as f:
            for rule, contradiction_types in rules:
                f.write(f"Rule: {rule}\n")
                f.write(f"Contradiction Types: {', '.join(contradiction_types)}\n\n")

    print(f"\n✅ Successfully saved effective rules with contradiction types in: {output_folder}")


# 主程序入口
def main():
    contradictions_file = '../dense_text_cross.txt'  # 替换为实际文件路径
    llm_folder = './dual_star/llm'  # 替换为实际文件夹路径
    output_folder = './dual_star/llm_evaluations_v2'  # 输出评估结果的文件夹路径

    # 加载矛盾数据
    contradictions = load_contradictions(contradictions_file)

    # 加载 llm 文件夹中的所有规则
    rules_by_file = load_rules_from_llm(llm_folder)

    # 存储有效规则及其对应的矛盾类型
    effective_rules = {}

    # 遍历规则并判断其是否有效
    for filename, rules in rules_by_file.items():
        effective_rules[filename] = []

        # 将所有规则转换为自然语言（一次性调用 LLM）
        rules_natural_language = convert_rules_to_natural_language(rules, llm)

        # 使用 LLM 判断规则与矛盾类型的匹配情况
        matching_results = rule_matches_contradictions_with_llm(rules_natural_language, contradictions, llm)

        # 解析匹配结果
        for rule, result in zip(rules, matching_results):
            matched_contradictions = []
            if '→' not in rule:
                continue  # 跳过不完整的规则
            if "match" in result.lower():
                # 提取匹配的矛盾类型（假设 LLM 给出了匹配的类型列表）
                matched_contradictions = re.findall(r'\b[A-Za-z0-9_]+\b', result)
            if matched_contradictions:
                effective_rules[filename].append((rule, matched_contradictions))

    # 保存有效规则及其对应的矛盾类型
    save_matched_rules(effective_rules, output_folder)


if __name__ == '__main__':
    main()
