import pandas as pd
import random
import os
import json
from typing import List, Dict, Any

# --- 配置 ---
file_path = 'snli_train.csv' # 假设输入文件仍为 CSV，但列名已修改
output_file_path = 'sample_data.json' # 更改输出文件名
sample_size = 1500

# 定义数字标签与文本标签的映射关系 (基于您的要求)
LABEL_MAPPING = {
    0: 'entailment',
    1: 'neutral', # 示例中的标签 '1'
    2: 'contradiction'
}
target_labels_text = list(LABEL_MAPPING.values())

# 输入数据的列名（基于您的示例）
PREMISE_COL = 'premise'
HYPOTHESIS_COL = 'hypothesis'
LABEL_COL = 'label' # 假设数字标签列名为 'label'

# 组合样本的设置
GROUP_SIZE = 10 # 10条原始样本组合成1条新样本
SEPARATOR = " [SEP] " # 用于连接 premise 和 hypothesis

# --- 数据加载与预处理 ---

print(f"--- 🚀 正在处理文件: {file_path} ---")

try:
    df = pd.read_csv(file_path)
    print(f"✅ 文件加载成功。原始样本数: {len(df)}")
except FileNotFoundError:
    print(f"错误：文件未找到在路径: {file_path}")
    exit()
except Exception as e:
    print(f"加载文件时发生错误: {e}")
    exit()

# 检查所需列是否存在
required_cols = [LABEL_COL, PREMISE_COL, HYPOTHESIS_COL]
if not all(col in df.columns for col in required_cols):
    print(f"错误：CSV 文件中缺少必需的列。期望列: {required_cols}")
    print(f"实际列: {df.columns.tolist()}")
    exit()


# ************ 关键修改 1: 标签映射和清理 ************

# 1. 确保标签列是整数，并清理无效行
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors='coerce').astype('Int64')
df = df.dropna(subset=[LABEL_COL]).copy()

# 2. 创建文本标签列
df['label_text'] = df[LABEL_COL].map(LABEL_MAPPING)

# 移除标签映射失败的行
df = df.dropna(subset=['label_text']).reset_index(drop=True)

# -----------------------------------------------------

# 存储采样后的数据
sampled_dfs = []

print("\n--- 正在进行分层采样 ---")

for label_text in target_labels_text:
    # 筛选出当前标签的所有样本
    subset = df[df['label_text'] == label_text]

    # 确保子集大小不小于所需采样数
    if len(subset) >= sample_size:
        sampled_subset = subset.sample(n=sample_size, random_state=42)
        sampled_dfs.append(sampled_subset)
    else:
        print(f"警告：标签 '{label_text}' 的总样本数 ({len(subset)}) 小于所需采样数 ({sample_size})。取全部样本。")
        sampled_dfs.append(subset)

# 合并所有采样的 DataFrame
combined_sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
print(f"✅ 完成采样。合并后总样本数: {len(combined_sampled_df)}")

# Step 2: 创建新的 text 字段（premise 和 hypothesis 组合）
combined_sampled_df['text'] = (
    combined_sampled_df[PREMISE_COL].astype(str) +
    SEPARATOR +
    combined_sampled_df[HYPOTHESIS_COL].astype(str)
)

# 对数据进行分组处理，确保每个标签组内进行重组
final_complex_samples = []

for label_text in target_labels_text:
    # 筛选出当前标签的全部采样样本
    group_df = combined_sampled_df[combined_sampled_df['label_text'] == label_text].copy().reset_index(drop=True)
    num_samples = len(group_df)

    expected_new_samples = num_samples // GROUP_SIZE

    print(f"\n--- 处理标签: **{label_text}** ---")
    print(f"原始样本数: {num_samples}，预期新样本数: {expected_new_samples}")

    # 循环进行样本组合
    for i in range(expected_new_samples):
        start_index = i * GROUP_SIZE
        end_index = start_index + GROUP_SIZE

        subset_to_combine = group_df.iloc[start_index:end_index]

        # 将这 GROUP_SIZE 个样本的 'text' 字段用空格连接起来
        new_text = " ".join(subset_to_combine['text'].astype(str).tolist())

        # 新样本的标签
        new_label_text = subset_to_combine['label_text'].iloc[0]
        new_label_int = subset_to_combine[LABEL_COL].iloc[0]

        # 创建新的样本记录
        new_sample = {
            'text': new_text,
            'label': new_label_int,      # 数字标签 (0, 1, 2)
            'label_text': new_label_text # 文本标签
        }
        final_complex_samples.append(new_sample)

# 将最终结果列表转换为 DataFrame
final_df = pd.DataFrame(final_complex_samples)

# 结果验证：
print("\n--- 最终复杂样本集信息 ---")
print(f"总样本数: **{len(final_df)}** (如果采样成功，应为 150*3 = 450)")
print("标签分布:")
print(final_df['label_text'].value_counts())

# ************ 关键修改在这里 ************
final_df.to_json(output_file_path, orient='records', force_ascii=False, indent=4)

print(f"\n✅ 处理完成，结果已保存到 JSON 文件: **{output_file_path}**")