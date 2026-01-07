"""
使用的数据集：RTE-3
使用的基准："xlnet-base-cased"
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
    EvalPrediction  # <--- 新增
)
from sklearn.metrics import accuracy_score
import os
import json

# --- 标签和文件定义 ---
xml_file = "../../../datasets/data_from_related_work/RTE/RTE3.xml"  # 请替换为实际路径

# 假设三元分类标签：
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

    valid_keys = LABEL_MAPPING.keys()

    for pair in root.findall('.//pair'):
        entailment = pair.get('entailment')

        if entailment in valid_keys:

            premise_elem = pair.find('t')
            hypothesis_elem = pair.find('h')

            premise = premise_elem.text.strip() if premise_elem is not None and premise_elem.text else ""
            hypothesis = hypothesis_elem.text.strip() if hypothesis_elem is not None and hypothesis_elem.text else ""

            if premise and hypothesis:
                data.append({
                    'premise': premise,
                    'hypothesis': hypothesis,
                    'label_str': entailment,
                    'label': LABEL_MAPPING[entailment]
                })

    df = pd.DataFrame(data)

    if df.empty:
        print("致命警告: 解析器返回空数据帧。请检查 XML 路径和标签！")

    return df


def preprocess_function(examples, tokenizer):
    """
    对句子对（premise, hypothesis）进行分词和编码。
    """
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


def compute_metrics(p: EvalPrediction):
    """计算准确率"""
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


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


# --- 均衡分层抽样函数 ---
def stratified_sample_split(df, target_size, samples_per_class, random_state=42):
    """按类别均衡抽取指定数量的样本，并返回剩余的样本。"""
    sampled_dfs = []

    for label_str in LABEL_MAPPING.keys():
        if label_str not in df['label_str'].unique():
            continue

        label_df = df[df['label_str'] == label_str]
        n_sample = min(len(label_df), samples_per_class)

        sampled_df = label_df.sample(n=n_sample, random_state=random_state)
        sampled_dfs.append(sampled_df)

    if not sampled_dfs:
        return pd.DataFrame(), df

    result_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)

    remaining_df = df[~df.index.isin(result_df.index)].reset_index(drop=True)

    return result_df.head(target_size), remaining_df


# --- 执行均衡抽样划分 ---

train_df, remaining_df = stratified_sample_split(
    full_df, N_TRAIN, SAMPLES_PER_CLASS_TRAIN
)

dev_df, remaining_df = stratified_sample_split(
    remaining_df, N_DEV, SAMPLES_PER_CLASS_DEV
)

test_df, _ = stratified_sample_split(
    remaining_df, N_TEST, SAMPLES_PER_CLASS_TEST
)

print("\n--- 均衡抽样划分结果 ---")
print(f"训练集 (目标{N_TRAIN}): {len(train_df)}")
print(f"验证集 (目标{N_DEV}): {len(dev_df)}")
print(f"测试集 (目标{N_TEST}): {len(test_df)}")
print("--- 训练集类别分布 ---")
print(train_df['label_str'].value_counts())

# --- 3. 转换为 Hugging Face Dataset ---

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

for ds in [train_dataset, validation_dataset, test_dataset]:
    if '__index_level_0__' in ds.column_names:
        ds = ds.remove_columns('__index_level_0__')

rte_datasets = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

print("数据集处理完毕，开始 XLNet 模型训练")

# --- 4. XLNet 训练流程 ---

MODEL_NAME_XLNET = "xlnet-base-cased"
OUTPUT_DIR_XLNET = "results/rte_tri_xlnet"

# 初始化分词器
tokenizer_xlnet = AutoTokenizer.from_pretrained(MODEL_NAME_XLNET)

# 应用预处理
print("--- 正在对数据集进行 XLNet 分词 ---")
tokenized_train_xlnet = train_dataset.map(lambda x: preprocess_function(x, tokenizer_xlnet), batched=True)
tokenized_val_xlnet = validation_dataset.map(lambda x: preprocess_function(x, tokenizer_xlnet), batched=True)
tokenized_test_xlnet = test_dataset.map(lambda x: preprocess_function(x, tokenizer_xlnet), batched=True)

# 初始化模型
model_xlnet = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_XLNET,
    num_labels=num_labels
)

# 定义训练参数
training_args_xlnet = TrainingArguments(
    output_dir=OUTPUT_DIR_XLNET,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir='./logs/rte_tri_xlnet',
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
    seed=42
)

# 初始化 Trainer
trainer_xlnet = Trainer(
    model=model_xlnet,
    args=training_args_xlnet,
    train_dataset=tokenized_train_xlnet,
    eval_dataset=tokenized_val_xlnet,
    tokenizer=tokenizer_xlnet,
    compute_metrics=compute_metrics
)

# 开始训练
print(f"\n--- 开始训练 XLNet ({MODEL_NAME_XLNET}) ---")
trainer_xlnet.train()

# 在测试集上评估最终模型
print("--- XLNet 在测试集上进行评估 ---")
metrics_xlnet = trainer_xlnet.evaluate(tokenized_test_xlnet)
print(f"XLNet 评估结果: {metrics_xlnet}")

# --- 保存评估结果到文件 ---
results_dir = OUTPUT_DIR_XLNET
os.makedirs(results_dir, exist_ok=True)
output_eval_file = os.path.join(results_dir, "eval_results_xlnet.json")

with open(output_eval_file, "w") as writer:
    print(f"--- 正在保存评估结果到 {output_eval_file} ---")
    json.dump(metrics_xlnet, writer, indent=4)

print("评估结果保存完成。")