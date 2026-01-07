import pandas as pd
import random

# 假设文件路径是正确的，并且文件编码是 UTF-8
file_path = 'mnli_train.csv'

# --- 警告：请确保 'mnli_train.csv' 文件存在于正确路径，否则代码将因 FileNotFoundError 终止 ---

try:
    # 尝试加载数据
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"错误：文件未找到在路径: {file_path}")
    # 实际操作中，您需要确保文件存在
    exit() # 如果文件不存在，停止执行
except Exception as e:
    print(f"加载文件时发生错误: {e}")
    exit()

# 定义目标标签和每个标签的采样数量
target_labels = ['contradiction', 'entailment', 'neutral']
sample_size = 1500

# 存储采样后的数据
sampled_dfs = []

for label in target_labels:
    # 筛选出当前标签的所有样本
    subset = df[df['label_text'] == label]

    # 确保子集大小不小于所需采样数
    if len(subset) >= sample_size:
        # 使用 sample() 方法进行随机采样，frac=None, n=sample_size
        # 使用 random_state 确保结果可复现
        sampled_subset = subset.sample(n=sample_size, random_state=42)
        sampled_dfs.append(sampled_subset)
    else:
        print(f"警告：标签 '{label}' 的总样本数 ({len(subset)}) 小于所需采样数 ({sample_size})。取全部样本。")
        sampled_dfs.append(subset)

# 合并所有采样的 DataFrame
combined_sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)

# Step 1: 创建新的 text 字段（使用特殊分隔符，以保持语义的独立性）
# 原始的 text1 和 text2 组合成新的 text 字段
combined_sampled_df['text'] = combined_sampled_df['text1'].astype(str) + " [SEP] " + combined_sampled_df['text2'].astype(str)

# 对数据进行分组处理，确保每个标签组内进行重组
final_complex_samples = []
group_size = 10  # 10条原始样本组合成1条新样本

for label in target_labels:
    # 筛选出当前标签的全部采样样本 (应该有1500条)
    group_df = combined_sampled_df[combined_sampled_df['label_text'] == label].copy().reset_index(drop=True)
    num_samples = len(group_df)

    # 预期新样本数量 (1500 / 10 = 150)
    expected_new_samples = num_samples // group_size

    print(f"\n--- 处理标签: **{label}** ---")
    print(f"原始样本数: {num_samples}，预期新样本数: {expected_new_samples}")


    # 循环进行样本组合
    for i in range(expected_new_samples):
        # 确定当前要组合的10个样本的索引范围
        start_index = i * group_size
        end_index = start_index + group_size

        # 提取这10个样本的子集
        subset_to_combine = group_df.iloc[start_index:end_index]

        # 将这10个样本的 'text' 字段用空格连接起来，形成一个新的复杂文本
        new_text = " ".join(subset_to_combine['text'].tolist())

        # 新样本的标签仍对应原始标签（这10条样本的标签相同）
        new_label = subset_to_combine['label_text'].iloc[0]

        # 创建新的样本记录
        new_sample = {
            'text': new_text,
            'label': subset_to_combine['label'].iloc[0],  # 保持原始标签（数字）
            'label_text': new_label
        }
        final_complex_samples.append(new_sample)

# 将最终结果列表转换为 DataFrame
final_df = pd.DataFrame(final_complex_samples)

# 结果验证：
print("\n--- 最终复杂样本集信息 ---")
print(f"总样本数: **{len(final_df)}** (符合 150*3 = 450)")
print("标签分布:")
print(final_df['label_text'].value_counts())

# ************ 关键修改在这里 ************
output_file_path = 'sample_data.json'
final_df.to_json(output_file_path, orient='records', force_ascii=False, indent=4)

print(f"\n✅ 处理完成，结果已保存到 JSON 文件: **{output_file_path}**")