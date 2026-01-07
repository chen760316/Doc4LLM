import json
import random
import os
import re
from typing import List, Dict, Any, Optional

# --- 配置 ---
INPUT_FILE = "train.txt"
OUTPUT_FILE = "sampled_data.json"  # 更改输出文件名以反映内容
SAMPLE_SIZE = 100
random.seed(42)  # 设置随机种子以保证结果可复现

# 定义用于连接 premise 和 hypothesis 的分隔符
SEPARATOR = " [HYPOTHESIS] "  # 保持不变


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

    # 假设可能存在的标签及其前缀
    possible_prefixes = ["non_entailment", "not_entailment"]  # 增加 'not_entailment'
    non_entailment_raw_texts = []

    # 1. 读取文件并筛选
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()

                found_prefix = None
                for prefix in possible_prefixes:
                    if stripped_line.startswith(prefix):
                        found_prefix = prefix
                        break

                if found_prefix:
                    # 移除前缀，并移除前缀后面的所有空白字符（包括空格和制表符），然后添加到列表中
                    content = stripped_line[len(found_prefix):].lstrip()
                    if content:  # 确保内容不为空
                        non_entailment_raw_texts.append(content)

    except Exception as e:
        print(f"🚨 读取文件或筛选时发生错误: {e}")
        return []

    total_available = len(non_entailment_raw_texts)
    print(f"✅ 成功找到 {total_available} 条 '{prefix}' 记录。")

    # 2. 采样
    num_to_sample = min(sample_size, total_available)
    if total_available == 0:
        print("⚠️ 警告：没有可用于采样的记录。")
        return []

    # 使用索引进行采样，以避免在内存中复制所有字符串
    sampled_indices = random.sample(range(total_available), num_to_sample)
    sampled_raw_texts = [non_entailment_raw_texts[i] for i in sampled_indices]

    print(f"✅ 实际采样 {len(sampled_raw_texts)} 条记录。")

    # 3. 解析并合并为 JSON 格式
    parsed_data: List[Dict[str, str]] = []

    # 假设文本主体和假设之间只有极少的空格，且假设通常以大写字母开头
    # 目标分隔模式: (句子结尾的标点后紧跟一个或多个空格) + (大写字母开头的假设)
    # 我们匹配最后一个句号/问号，并将其后的内容视作假设
    # 正则表达式：查找紧接在句号或问号后面的空格和大写字母
    # 注意：使用非贪婪匹配 (.*?) 来匹配主体，直到最后一个分隔符
    # 修正后的匹配逻辑：尝试找到最右边的分隔符（.或?）后面跟着一个或多个空格，然后是一个大写字母
    # 考虑到数据格式，我们使用最简单的分割方法：
    # 找到最后一个以标点符号（.或?）结尾的主体，假设它后面的所有内容都是 hypothesis

    # 匹配模式：(主体内容) + (标点符号) + (1+个空格) + (假设内容)
    # 示例中的数据没有明确的分隔符，我们必须依赖最后一个标点作为分割的提示。

    for line in sampled_raw_texts:

        # 尝试通过找到最后一个句子的末尾标点（. 或 ?）来分割主体和假设
        # 匹配 `[.,?]` 后面跟着 `\s+`（一个或多个空格），再跟着一个大写字母 `[A-Z]`
        match = re.search(r'[.?!]\s+([A-Z].*)', line)

        story_text = ""
        hypothesis = ""

        if match:
            # 找到假设开始的位置
            # match.start() 是整个模式开始的位置 (标点符号的位置)
            # match.group(0) 是整个匹配字符串 (标点到假设结尾)

            # 假设的内容是 group(1)
            hypothesis = match.group(1).strip()

            # 故事主体是假设之前的所有内容，包括假设之前的那个标点符号
            # 我们可以通过找到 match.start() 来确定分割点
            split_index = line.find(match.group(1))

            # 主体是开头到假设开始之前的文本
            # 我们需要保留 hypothesis 之前的标点符号作为主体的一部分。
            # 假设主体和假设之间只有一个空格，我们从标点符号前找分割点。

            # 重新计算分割点：找到最后一个标点符号的位置，然后用它分割
            # 找到最后一个句号或问号的位置
            last_period = line.rfind('.')
            last_question = line.rfind('?')
            last_excl = line.rfind('!')

            last_punc_index = max(last_period, last_question, last_excl)

            if last_punc_index != -1 and last_punc_index < len(line) - 5:  # 确保标点不在末尾，且后面有足够长的假设
                # 分割点在标点符号之后，大写字母之前
                # 我们认为假设从紧跟在标点后面的第一个大写字母开始

                # 再次使用正则表达式来精确寻找分割索引
                # 模式：(一个标点符号) + (1个或多个空格) + (大写字母)
                split_match = re.search(r'[.?!](\s+)([A-Z])', line)

                if split_match:
                    # 分割点：标点符号的位置 + 标点符号的长度 (1) + 后面空格的长度
                    # story_text 包含标点符号
                    story_text = line[:split_match.start() + 1].strip()

                    # hypothesis 从大写字母开始
                    hypothesis = line[split_match.start() + len(split_match.group(0)) - len(split_match.group(2)):].strip()
                else:
                    # 如果复杂模式分割失败，使用原始的 match 来做粗略分割
                    story_text = line[:match.start()].strip()
                    hypothesis = match.group(1).strip()

            else:
                # 如果没有找到合适的分割点，将整行视为 text (作为兜底)
                story_text = line
                hypothesis = ""

        else:
            # 如果没有找到任何分割模式，则将整行视为主体，假设为空
            story_text = line
            hypothesis = ""

        # 4. 核心：组合
        if hypothesis:
            merged_text = f"{story_text}{SEPARATOR}{hypothesis}"
        else:
            # 如果分割失败，只保留原始文本
            merged_text = story_text

        parsed_data.append({
            "label": "not_entailment",
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
    # 注意：由于原始数据格式的模糊性，分割可能不完美，这取决于实际 train.txt 中分隔符的稳定程度。
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
    # 确保 re 模块被导入
    main()