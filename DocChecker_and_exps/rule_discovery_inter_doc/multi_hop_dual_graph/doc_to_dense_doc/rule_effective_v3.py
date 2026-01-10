import os
import json
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict

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

def extract_contradiction_positions(contradictions: List[Dict[str, str]], llm: OpenAIClient) -> List[Dict[str, str]]:
    """
    提取矛盾位置或类型
    :param contradictions: 包含矛盾信息的列表
    :param llm: OpenAIClient 实例，用于与 LLM 交互
    :return: 更新后的矛盾列表，包含每个矛盾的 'contradiction_position'
    """
    prompt = "For each contradiction reason below, please identify the location or specific aspect of the contradiction (e.g., 'Numeric', 'Temporal', 'Emotional', etc.).\n\n"

    # 生成每个矛盾的说明
    for contradiction in contradictions:
        prompt += f"Contradiction: {contradiction['reason']}\n"

    # 提交给 LLM 提取矛盾位置
    response = llm.generate(
        user_message=prompt,
        system_message="You are an expert in contradiction analysis. Your task is to extract the specific location or type of contradiction from the provided reasons."
    )

    # 解析 LLM 返回的结果
    positions = response.strip().split('\n')

    # 更新每个矛盾项
    for i, contradiction in enumerate(contradictions):
        contradiction['contradiction_position'] = positions[i].strip()  # 将提取的矛盾位置添加到数据中

    return contradictions

def read_contradictions_from_file(file_path: str) -> List[Dict[str, str]]:
    """
    从指定文件读取矛盾描述
    :param file_path: 文件路径
    :return: 包含矛盾描述的列表
    """
    contradictions = []

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()

        # 将每个文件内容作为一个矛盾描述，封装为字典形式
        contradiction = {
            'file_name': os.path.basename(file_path),
            'reason': content
        }
        contradictions.append(contradiction)

    return contradictions

def read_rules_from_files(input_dir: str) -> Dict[str, List[str]]:
    """
    读取指定文件夹下的规则文件，并返回规则列表
    :param input_dir: 存储规则数据的文件夹路径
    :return: 每个文件包含的规则字典
    """
    rules_by_file = {}

    # 遍历目录下的所有 txt 文件
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(input_dir, file_name)

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()

                # 假设规则以行分隔
                rules = content.split('\n')
                rules_by_file[file_name] = [rule.strip() for rule in rules if rule.strip()]

    return rules_by_file


def refine_contradiction_positions_with_rules(contradictions: List[Dict[str, str]], rules_by_file: Dict[str, List[str]], llm: OpenAIClient) -> List[Dict[str, str]]:
    """
    使用规则润色矛盾位置，同时判断哪些规则反映矛盾
    :param contradictions: 包含矛盾信息的列表
    :param rules_by_file: 每个文件的规则字典
    :param llm: OpenAIClient 实例，用于与 LLM 交互
    :return: 更新后的矛盾列表，包含润色后的 'contradiction_position' 和反映矛盾的规则
    """
    for contradiction in contradictions:
        prompt = f"Refine the contradiction position for the following contradiction, considering the rules provided:\n"
        prompt += f"Contradiction: {contradiction['reason']}\n"
        prompt += f"Contradiction Position: {contradiction['contradiction_position']}\n"
        prompt += "Please provide a more refined position based on the given rules.\n\n"

        # 添加所有相关规则
        rules = rules_by_file.get(contradiction['file_name'], [])

        # 检查并确保所有规则都被列出
        if not rules:
            prompt += "No rules found for this file.\n"
        else:
            prompt += "Rules:\n"
            for rule in rules:
                prompt += f"- {rule}\n"

        # 生成一个子问题，询问哪些规则反映了矛盾
        prompt += "\nPlease identify which of these rules reflect the contradiction. List the rules that match the contradiction:\n"

        # 提交给 LLM 进行润色和规则匹配
        response = llm.generate(
            user_message=prompt,
            system_message="You are an expert in contradiction analysis. Refine the contradiction position based on the provided rules, and identify which rules reflect the contradiction."
        )

        # 解析 LLM 的返回，获取反映矛盾的规则
        refined_position_and_rules = response.strip().split('\n')

        # 将返回的结果分为润色后的矛盾位置和反映矛盾的规则
        refined_position = refined_position_and_rules[0].strip()
        reflecting_rules = refined_position_and_rules[1:]

        # 更新矛盾数据
        contradiction['contradiction_position'] = refined_position
        contradiction['reflecting_rules'] = reflecting_rules  # 保存反映矛盾的规则

    return contradictions



# 主程序入口
if __name__ == "__main__":
    # 读取规则文件
    rules_by_file = read_rules_from_files(INPUT_DIR)

    # 遍历所有输入文件夹中的 txt 文件
    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith('.txt'):
            file_path = os.path.join(INPUT_DIR, file_name)

            # 从文件中读取矛盾数据
            contradictions = read_contradictions_from_file(file_path)

            # 提取矛盾位置
            updated_contradictions = extract_contradiction_positions(contradictions, llm)

            # 对矛盾位置进行润色
            refined_contradictions = refine_contradiction_positions_with_rules(updated_contradictions, rules_by_file, llm)

            # 为每个文件创建独立的输出文件
            output_file_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}_refined.json")

            # 将更新后的数据保存到文件中
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(refined_contradictions, f, ensure_ascii=False, indent=2)

            print(f"数据已保存到 {output_file_path}")
