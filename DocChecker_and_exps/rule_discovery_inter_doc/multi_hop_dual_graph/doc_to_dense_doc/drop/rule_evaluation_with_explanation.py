"""
对rule_evaluation.py得到的有效规则进行进一步的解释，结果保存在llm_generations_explanations文件夹下
"""
import os
import re
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient

# 设定你的目录
LLM_GENERATIONS_DIR = './dual_star/llm_evaluations'
EXPLANATION_DIR = '../dual_star/llm_generations_explanations'

os.makedirs(EXPLANATION_DIR, exist_ok=True)

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

# 1. 读取 llm_generations 文件夹中的规则文件
def load_rules_from_llm_generations(folder_path):
    rules_by_file = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                rules = content.strip().split("\n")
                # 去除空行
                rules_by_file[filename] = [rule.strip() for rule in rules if rule.strip()]

    return rules_by_file


# 2. 构建解释规则的提示
def build_explanation_prompt(rule):
    prompt = f"Please provide a detailed explanation of why the following rule is effective in contradiction detection.\n"
    prompt += f"Rule: {rule}\n"
    prompt += "### Task:\nExplain why this rule is effective in contradiction detection. Provide details about the rule's reasoning."
    return prompt


# 3. 获取规则解释
def get_explanation_from_llm(rule, llm):
    prompt = build_explanation_prompt(rule)

    system_message = (
        "You are a contradiction detection assistant. Please provide a detailed explanation of why the rule is effective in contradiction detection."
    )

    # 提交给 LLM 进行解释
    response = llm.generate(
        user_message=prompt,
        system_message=system_message
    )

    explanation = response.strip()
    return explanation


# 4. 保存规则及其解释到文件
def save_rule_with_explanation(rule, explanation, filename):
    output_filename = os.path.join(EXPLANATION_DIR, filename)

    with open(output_filename, "a", encoding="utf-8") as f:
        f.write(f"Rule: {rule}\n")
        f.write(f"Explanation: {explanation}\n\n")

    print(f"✅ Saved explanation for rule in {output_filename}")


# 5. 主程序入口
def main():
    # 加载 llm_generations 文件夹中的所有规则
    rules_by_file = load_rules_from_llm_generations(LLM_GENERATIONS_DIR)

    # 对每个规则生成解释并保存
    for filename, rules in rules_by_file.items():
        for rule in rules:
            # 获取 LLM 对规则的解释
            explanation = get_explanation_from_llm(rule, llm)
            # 保存规则和解释
            save_rule_with_explanation(rule, explanation, filename)


if __name__ == '__main__':
    main()
