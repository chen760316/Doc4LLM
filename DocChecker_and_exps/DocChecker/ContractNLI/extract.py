import pandas as pd
import json
import random
import os
from typing import Dict, Any, List

# --- 配置 ---
INPUT_FILE = "contractnli_a_train.csv"
OUTPUT_FILE = "sampled_data.json"
SAMPLE_SIZE = 100  # 目标采样数量
random.seed(42)  # 设置随机种子以保证结果可复现

# 定义用于连接 premise 和 hypothesis 的分隔符
SEPARATOR = " [HYPOTHESIS] "

# 定义标签映射
LABEL_MAPPING = {
    0: "contradiction",
    1: "entailment",
    2: "neutral"
}


# --- 数据处理函数 ---

def sample_and_process_data(file_path: str, sample_size: int) -> List[Dict[str, Any]]:
    """
    加载 CSV 文件，采样指定数量的样本，并合并 premise 和 hypothesis。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return []

    print(f"--- 🚀 正在读取和采样文件: {file_path} ---")

    try:
        # 1. 使用 pandas 读取 CSV 文件
        # 假设文件是 UTF-8 编码
        df = pd.read_csv(file_path, encoding='utf-8')

        # 确保关键列存在
        required_cols = ['premise', 'hypothesis', 'label']
        if not all(col in df.columns for col in required_cols):
            print(f"🚨 错误：CSV 文件中缺少必需的列 ('premise', 'hypothesis', 'label')。")
            return []

        # 2. 随机采样
        total_available = len(df)
        num_to_sample = min(sample_size, total_available)

        if total_available == 0:
            print("⚠️ 警告：文件中没有数据可供采样。")
            return []

        # 执行采样
        sampled_df = df.sample(n=num_to_sample, random_state=42)

        print(f"✅ 总可用样本数: {total_available}")
        print(f"✅ 实际采样数: {num_to_sample}")

        # 3. 合并和格式化
        processed_list = []
        for index, row in sampled_df.iterrows():
            premise = str(row['premise']).strip()
            hypothesis = str(row['hypothesis']).strip()
            label_int = int(row['label'])

            # 构造合并后的 text 字段
            merged_text = f"{premise}{SEPARATOR}{hypothesis}"

            processed_list.append({
                "label": LABEL_MAPPING.get(label_int, "unknown"),  # 使用映射后的文本标签
                "label_int": label_int,  # 保留原始数字标签
                "text": merged_text
            })

        return processed_list

    except Exception as e:
        print(f"🚨 处理文件时发生错误: {e}")
        return []


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
    # 1. 加载、采样和处理数据
    sampled_data = sample_and_process_data(INPUT_FILE, SAMPLE_SIZE)

    if not sampled_data:
        return

    # 2. 保存结果
    save_json_data(sampled_data, OUTPUT_FILE)

    # 打印预览
    print("\n[采样数据预览 (第一个实例)]:")
    if sampled_data:
        print(json.dumps(sampled_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()