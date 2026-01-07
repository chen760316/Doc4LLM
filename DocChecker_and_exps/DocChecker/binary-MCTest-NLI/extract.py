import json
import random
import os
import re
from typing import List, Dict, Any

# --- 配置 ---
INPUT_FILE = "data.txt"
OUTPUT_FILE = "sampled_data.json"
SAMPLE_SIZE = 100
random.seed(42)  # 设置随机种子以保证结果可复现


# --- 数据处理函数 ---

def extract_and_sample_non_entailment(file_path: str, sample_size: int) -> List[Dict[str, str]]:
    """
    加载文本文件，提取 'non_entailment' 行，并进行采样。
    将每行文本解析为 (text + hypothesis) 并合并到单一的 'text' 字段中。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return []

    print(f"--- 🚀 正在读取和筛选文件: {file_path} ---")

    non_entailment_lines = []

    # 1. 读取文件并筛选
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line.startswith("non_entailment"):
                    # 移除前缀，并添加到列表中
                    non_entailment_lines.append(stripped_line[len("non_entailment"):].strip())
    except Exception as e:
        print(f"🚨 读取文件或筛选时发生错误: {e}")
        return []

    total_available = len(non_entailment_lines)
    print(f"✅ 成功找到 {total_available} 条 'non_entailment' 记录。")

    # 2. 采样
    num_to_sample = min(sample_size, total_available)
    if total_available == 0:
        print("⚠️ 警告：没有可用于采样的记录。")
        return []

    sampled_lines = random.sample(non_entailment_lines, num_to_sample)
    print(f"✅ 实际采样 {len(sampled_lines)} 条记录。")

    # 3. 解析并合并为 JSON 格式
    parsed_data: List[Dict[str, str]] = []

    # 定义一个自定义分隔符用于合并，确保后续处理可以识别原始结构
    MERGE_SEPARATOR = " [HYPOTHESIS] "

    for line in sampled_lines:
        # 尝试将第一个大写字母开头的句子作为假设，但这次是为了准确分割后合并
        # 匹配模式: (故事主体 + 句号或问号) + 1个或多个空格 + (大写字母开头的假设)
        match = re.search(r'(\.\s*|\?\s*)(\s+([A-Z].*))', line)

        if match and len(line) > 10:  # 确保找到匹配且行不是太短
            # 故事主体 (Text)
            # 结束位置是 match.start() + 匹配到的第一个分组的长度
            text_end_index = line.find(match.group(2))
            story_text = line[:text_end_index].strip()

            # 假设 (Hypothesis)
            hypothesis = match.group(3).strip()

            # 🎯 核心修正: 合并 text 和 hypothesis
            merged_text = f"{story_text}{MERGE_SEPARATOR}{hypothesis}"
        else:
            # 如果分割失败，将整行视为合并后的文本
            merged_text = line

        parsed_data.append({
            "label": "non_entailment",
            "text": merged_text
        })

    return parsed_data


def save_json_data(data: List[Dict[str, Any]], filename: str):
    """保存 JSON 格式的数据到指定文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 以便正确保存中文等非ASCII字符
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 成功保存 {len(data)} 条数据到 {filename}")
    except Exception as e:
        print(f"🚨 错误：保存文件 {filename} 失败: {e}")


def main():
    # 1. 提取和采样数据
    sampled_data = extract_and_sample_non_entailment(INPUT_FILE, SAMPLE_SIZE)

    if not sampled_data:
        print("任务终止：没有数据可供保存。")
        return

    # 2. 保存结果
    save_json_data(sampled_data, OUTPUT_FILE)

    # 打印预览
    print("\n[结构化采样数据预览 (前 2 个实例)]:")
    print(json.dumps(sampled_data[:2], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import re  # 确保 re 模块被导入

    main()