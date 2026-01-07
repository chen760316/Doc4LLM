import pandas as pd
import json
import os
import random
from typing import Dict, Any

# --- 配置 ---
INPUT_CSV_FILE = "sampled_train.csv"
OUTPUT_JSON_FILE = "sampled_data.json"
TOTAL_SAMPLES = 500
SAMPLES_PER_CLASS = TOTAL_SAMPLES // 2  # 250


def sample_and_format_data(input_file: str, output_file: str, target_count: int) -> None:
    """
    从 CSV 文件中按比例采样数据，并格式化为要求的 JSON 结构。

    Args:
        input_file (str): 输入 CSV 文件路径。
        output_file (str): 输出 JSON 文件路径。
        target_count (int): 总共采样的样本数量。
    """
    samples_per_class = target_count // 2

    # 1. 读取 CSV 文件
    if not os.path.exists(input_file):
        print(f"🚨 错误：输入文件未找到: {input_file}")
        return

    try:
        # 尝试读取 CSV 文件
        df = pd.read_csv(input_file)
        print(f"✅ 成功读取文件: {input_file}，总行数: {len(df)}")
    except Exception as e:
        print(f"🚨 错误：读取 CSV 文件失败: {e}")
        return

    # 2. 按 is_correct 字段分组
    df_correct = df[df['is_correct'] == 1]
    df_incorrect = df[df['is_correct'] == 0]

    print(f"   - is_correct=1 样本数 (neg): {len(df_correct)}")
    print(f"   - is_correct=0 样本数 (pos): {len(df_incorrect)}")

    # 3. 随机采样
    # 使用 min() 确保不会采样超过现有样本数
    sample_count_correct = min(samples_per_class, len(df_correct))
    sample_count_incorrect = min(samples_per_class, len(df_incorrect))

    df_sample_correct = df_correct.sample(n=sample_count_correct, random_state=42)
    df_sample_incorrect = df_incorrect.sample(n=sample_count_incorrect, random_state=42)

    print(f"   - 实际采样 is_correct=1 (neg) 样本数: {sample_count_correct}")
    print(f"   - 实际采样 is_correct=0 (pos) 样本数: {sample_count_incorrect}")

    # 4. 格式化数据并构建 JSON 结构
    final_data: Dict[str, Dict[str, Any]] = {
        "pos": {},  # is_correct = 0 (负样本，即错误的答案/不正确的引用)
        "neg": {}  # is_correct = 1 (正样本，即正确的答案/正确的引用)
    }

    def format_row(row: pd.Series, doc_type: str) -> Dict[str, str]:
        """根据要求格式化单个行数据。"""
        # 组成要求的 text 字段
        full_text = f"{row['citing_prompt']}\n{row['answer_text']}"

        # 构造最终的结构，使用 example_id 作为键
        return {
            "text": full_text,
            "is_correct": row['is_correct'],
            "doc_type": doc_type
        }

    # 处理 is_correct = 0 的样本 (pos)
    for index, row in df_sample_incorrect.iterrows():
        # 使用 example_id 作为唯一键
        unique_id = str(row['example_id'])
        final_data["pos"][unique_id] = format_row(row, "pos")

    # 处理 is_correct = 1 的样本 (neg)
    for index, row in df_sample_correct.iterrows():
        unique_id = str(row['example_id'])
        final_data["neg"][unique_id] = format_row(row, "neg")

    # 5. 保存为 JSON 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 确保中文字符正确显示
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 成功保存数据到: {output_file}")
        print(f"   - JSON 文件包含的总样本数: {len(final_data['pos']) + len(final_data['neg'])}")
    except Exception as e:
        print(f"🚨 错误：保存 JSON 文件失败: {e}")


# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 在实际运行中，请确保 sampled_train.csv 文件在当前目录下或使用正确的路径
    # 示例用法：
    sample_and_format_data(INPUT_CSV_FILE, OUTPUT_JSON_FILE, TOTAL_SAMPLES)