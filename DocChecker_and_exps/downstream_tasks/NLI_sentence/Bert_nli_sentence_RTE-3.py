"""
使用的数据集：RTE-3
使用的基准："bert-base-uncased"
"""
import pandas as pd
import xml.etree.ElementTree as ET
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction # <--- 新增
)
from sklearn.metrics import accuracy_score
import os
import json

# --- 标签和文件定义 ---
xml_file = "../../../datasets/data_from_related_work/RTE/RTE3.xml"  # 请替换为实际路径

# 假设三元分类标签：YES (Entailment), CONTRADICTION, UNKNOWN (Neutral)
# 注意：RTE3的原始XML只有YES/NO，但我们尊重您的三元分类需求。
LABEL_MAPPING = {"ENTAILMENT": 0, "CONTRADICTION": 1, "UNKNOWN": 2}
num_labels = len(LABEL_MAPPING)

# --- 抽取数量定义 ---
N_TRAIN = 1500
N_DEV = 500
N_TEST = 500

# 期望每个数据集的每个类别的样本数
SAMPLES_PER_CLASS_TRAIN = N_TRAIN // num_labels
SAMPLES_PER_CLASS_DEV = N_DEV // num_labels
SAMPLES_PER_CLASS_TEST = N_TEST // num_labels


def parse_rte_xml(xml_path, LABEL_MAPPING):
    """
    解析 RTE-like XML 文件，直接读取 ENTALIMENT, CONTRADICTION, UNKNOWN 标签。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []

    # 直接使用 XML 中的 entailment 属性作为 standard_label
    valid_keys = LABEL_MAPPING.keys()

    for pair in root.findall('.//pair'):
        entailment = pair.get('entailment')  # e.g., "ENTAILMENT", "CONTRADICTION", "UNKNOWN"

        # 1. 检查标签是否是我们预期的三元标签之一
        if entailment in valid_keys:

            premise_elem = pair.find('t')
            hypothesis_elem = pair.find('h')

            premise = premise_elem.text.strip() if premise_elem is not None and premise_elem.text else ""
            hypothesis = hypothesis_elem.text.strip() if hypothesis_elem is not None and hypothesis_elem.text else ""

            # 2. 检查 premise 和 hypothesis 是否有效
            if premise and hypothesis:
                data.append({
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'label_str': entailment,
                    'label': LABEL_MAPPING[entailment]
                })

    df = pd.DataFrame(data)

    # 返回前，再次确保没有空数据
    if df.empty:
        print("致命警告: 解析器返回空数据帧。请检查 XML 路径和标签！")

    return df


# --- 2. 数据加载、清洗和均衡抽取 ---

print("--- 正在加载并解析 RTE XML 文件 ---")
try:
    full_df = parse_rte_xml(xml_file, LABEL_MAPPING)
    print(f"解析成功，总计有效样本数: {len(full_df)}")
except Exception as e:
    print(f"加载或解析文件时发生错误: {e}")
    exit()

# 检查可用的类别
available_labels = full_df['label_str'].unique()
if len(available_labels) < num_labels:
    print(f"🚨 警告: 无法实现三元均衡抽取，因为数据中只包含 {available_labels} 标签。")

def preprocess_function(examples, tokenizer):
    """
    对句子对（premise, hypothesis）进行分词和编码。
    """
    # NLI是句子对分类任务，输入为 (premise, hypothesis)
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=128 # 推荐长度
    )


# --- 均衡分层抽样函数 ---
def stratified_sample_split(df, target_size, samples_per_class, random_state=42):
    """按类别均衡抽取指定数量的样本，并返回剩余的样本。"""
    sampled_dfs = []

    for label_str in LABEL_MAPPING.keys():
        # 确保每个类别都有足够的样本
        if label_str not in df['label_str'].unique():
            continue

        label_df = df[df['label_str'] == label_str]
        n_sample = min(len(label_df), samples_per_class)  # 确保不超界

        # 抽样
        sampled_df = label_df.sample(n=n_sample, random_state=random_state)
        sampled_dfs.append(sampled_df)

    if not sampled_dfs:
        return pd.DataFrame(), df

    result_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 找出未被抽样的剩余样本
    remaining_df = df[~df.index.isin(result_df.index)].reset_index(drop=True)

    return result_df.head(target_size), remaining_df

# 评估指标函数 (修复: compute_metrics 定义)
def compute_metrics(p: EvalPrediction): # <--- 使用 EvalPrediction
    """计算准确率"""
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# --- 执行均衡抽样划分 ---

# 1. 抽取训练集
train_df, remaining_df = stratified_sample_split(
    full_df, N_TRAIN, SAMPLES_PER_CLASS_TRAIN
)

# 2. 从剩余样本中抽取开发集 (Dev/Validation)
dev_df, remaining_df = stratified_sample_split(
    remaining_df, N_DEV, SAMPLES_PER_CLASS_DEV
)

# 3. 从剩余样本中抽取测试集
test_df, _ = stratified_sample_split(
    remaining_df, N_TEST, SAMPLES_PER_CLASS_TEST
)

print("\n--- 均衡抽样划分结果 ---")
print(f"训练集 (目标{N_TRAIN}): {len(train_df)}")
print(f"验证集 (目标{N_DEV}): {len(dev_df)}")
print(f"测试集 (目标{N_TEST}): {len(test_df)}")
print("--- 训练集类别分布 ---")
print(train_df['label_str'].value_counts())

# --- 3. 转换为 Hugging Face Dataset 并继续训练流程 ---

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

# 删除由 pandas 索引转换产生的冗余列（如果存在）
for ds in [train_dataset, validation_dataset, test_dataset]:
    if '__index_level_0__' in ds.column_names:
        ds = ds.remove_columns('__index_level_0__')

rte_datasets = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

print("数据集处理完毕，开始Bert模型训练")

# --- 4. BERT 训练流程 (与原代码一致，但需注意 num_labels 必须为 3) ---

MODEL_NAME_BERT = "bert-base-uncased"
OUTPUT_DIR_BERT = "results/rte_tri_bert"  # 更改输出目录名

tokenizer_bert = AutoTokenizer.from_pretrained(MODEL_NAME_BERT)

# 应用预处理
tokenized_train_bert = train_dataset.map(lambda x: preprocess_function(x, tokenizer_bert), batched=True)
tokenized_val_bert = validation_dataset.map(lambda x: preprocess_function(x, tokenizer_bert), batched=True)

# 初始化模型
# 🚨 注意：如果您的数据实际只有两类，但这里 num_labels=3，模型会训练失败或结果无意义。
model_bert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_BERT,
    num_labels=num_labels  # 使用定义的 num_labels (假设为 3)
)

# 定义训练参数 (与原代码一致)
training_args_bert = TrainingArguments(
    output_dir=OUTPUT_DIR_BERT,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir='./logs/rte_tri_bert',
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
    seed=42
)

# 初始化 Trainer
trainer_bert = Trainer(
    model=model_bert,
    args=training_args_bert,
    train_dataset=tokenized_train_bert,
    eval_dataset=tokenized_val_bert,
    tokenizer=tokenizer_bert,
    compute_metrics=compute_metrics
)

# 开始训练
print(f"\n--- 开始训练 BERT ({MODEL_NAME_BERT}) ---")
trainer_bert.train()

# 在测试集上评估最终模型
tokenized_test_bert = test_dataset.map(lambda x: preprocess_function(x, tokenizer_bert), batched=True)
print("--- BERT 在测试集上进行评估 ---")
metrics_bert = trainer_bert.evaluate(tokenized_test_bert)
print(f"BERT 评估结果: {metrics_bert}")

# --- 保存评估结果到文件 ---
results_dir = OUTPUT_DIR_BERT
os.makedirs(results_dir, exist_ok=True)
output_eval_file = os.path.join(results_dir, "eval_results.json")

with open(output_eval_file, "w") as writer:
    print(f"--- 正在保存评估结果到 {output_eval_file} ---")
    json.dump(metrics_bert, writer, indent=4)

print("评估结果保存完成。")