import pandas as pd  # 不再需要，但保留以防万一
import json
import random
import os
from typing import Dict, Any, List, Optional

# --- 配置 ---
INPUT_FILE = "../../../../../../datasets/data_from_related_work/DOCNLI/Data/DocNLI/train.json"  # 更改为目标输入文件
OUTPUT_FILE = "sampled_data.json"
SAMPLE_SIZE = 20  # 目标采样数量
TARGET_LABEL = "not_entailment"  # 新的目标标签
random.seed(42)  # 设置随机种子以保证结果可复现

# 定义用于连接 premise 和 hypothesis 的分隔符
SEPARATOR = " [HYPOTHESIS] "


# --- 数据处理函数 ---

def load_json_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    尝试从 JSON 文件中加载数据。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return None

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"✅ 文件结构为列表。总实例数: {len(data)}")
            return data
        else:
            print("🚨 错误：JSON 文件内容不是预期的列表格式。")
            return None

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None


def sample_and_process_data(all_data: List[Dict[str, Any]], sample_size: int, target_label: str) -> List[Dict[str, Any]]:
    """
    筛选出指定标签的样本，采样指定数量，并合并 premise 和 hypothesis。
    """
    # 1. 筛选目标标签样本
    filtered_list = [
        item for item in all_data
        if item.get('label') == target_label
    ]

    total_available = len(filtered_list)
    num_to_sample = min(sample_size, total_available)

    if total_available == 0:
        print(f"⚠️ 警告：未找到任何 '{target_label}' 标签的样本。")
        return []

    # 2. 随机采样
    # random.sample() 自动处理了随机状态和不重复采样
    sampled_list = random.sample(filtered_list, num_to_sample)

    print(f"✅ 可用 '{target_label}' 样本数: {total_available}")
    print(f"✅ 实际采样数: {num_to_sample}")

    # 3. 合并和格式化
    processed_list = []
    for item in sampled_list:
        premise = str(item.get('premise', '')).strip()
        hypothesis = str(item.get('hypothesis', '')).strip()

        # 确保关键字段存在
        if not premise or not hypothesis:
            continue

        # 构造合并后的 text 字段
        merged_text = f"{premise}{SEPARATOR}{hypothesis}"

        processed_list.append({
            "label": target_label,
            "text": merged_text
        })

    return processed_list


def save_json_data(data: List[Dict[str, Any]], filename: str):
    """保存 JSON 格式的数据到指定文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 以便正确保存中文等非ASCII字符
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 成功保存 {len(data)} 条样本到 {filename}")
    except Exception as e:
        print(f"🚨 错误：保存文件 {filename} 失败: {e}")


def main():
    # 1. 加载数据
    all_data = load_json_data(INPUT_FILE)
    if all_data is None:
        return

    # 2. 筛选、采样和处理数据
    # 注意：这里将目标标签设置为 "not_entailment"
    sampled_data = sample_and_process_data(all_data, SAMPLE_SIZE, TARGET_LABEL)

    if not sampled_data:
        return

    # 3. 保存结果
    save_json_data(sampled_data, OUTPUT_FILE)

    # 打印预览
    print("\n[采样数据预览 (第一个实例)]:")
    if sampled_data:
        print(json.dumps(sampled_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()