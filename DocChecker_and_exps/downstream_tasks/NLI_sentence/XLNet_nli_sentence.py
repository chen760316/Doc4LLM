"""
使用的数据集：SNLI
使用的基准："xlnet-base-cased"
"""
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction # <--- 新增
)
import numpy as np
from sklearn.metrics import accuracy_score
import os

# --- 1. 关键变量和函数定义 ---

# 定义文件路径
# 请注意：如果您运行代码时文件未找到，请检查这些路径是否正确。
train_file = "../../datasets/data_from_related_work/NLI/SNLI/snli_train.csv"
validation_file = "../../datasets/data_from_related_work/NLI/SNLI/snli_validation.csv"
test_file = "../../datasets/data_from_related_work/NLI/SNLI/snli_test.csv"

# NLI 任务的有效整数标签
valid_labels_int = [0, 1, 2]
num_labels = len(valid_labels_int)  # 仍然是 3


# 数据清洗函数 (适配整数标签)
def clean_df(df):
    """删除标签缺失和句子缺失的行。"""
    if 'label' not in df.columns:
        raise ValueError("DataFrame中缺少'label'列，请检查您的CSV文件！")

    # 确保标签是整数类型，方便过滤
    try:
        df['label'] = df['label'].astype(int)
    except Exception:
        # 如果转换失败，可能是数据混乱，过滤掉非数字项
        print("警告：标签列包含非整数值，将尝试跳过。")

    # 过滤掉标签不在有效整数列表中的行
    df = df[df['label'].isin(valid_labels_int)].copy()

    # 删除前提或假设缺失的行
    df.dropna(subset=['premise', 'hypothesis'], inplace=True)

    return df

# 预处理函数 (修复: preprocess_function 定义)
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

# 评估指标函数 (修复: compute_metrics 定义)
def compute_metrics(p: EvalPrediction): # <--- 使用 EvalPrediction
    """计算准确率"""
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


# --- 2. 数据加载、清洗和转换 ---

print("--- 正在加载本地 CSV 文件 ---")
try:
    # 限制读取的行数
    N_ROWS = 1000
    # 限制只读取前 N_ROWS 行
    train_df = pd.read_csv(train_file, nrows=N_ROWS)
    validation_df = pd.read_csv(validation_file, nrows=N_ROWS)
    test_df = pd.read_csv(test_file, nrows=N_ROWS)
    print(f"文件加载成功！每个数据集已限制为 {N_ROWS} 行。")
except FileNotFoundError as e:
    print(f"错误：文件未找到。请检查路径：{e.filename}")
    exit()
except Exception as e:
    print(f"加载文件时发生其他错误: {e}")
    exit()

# 数据清洗
# 注意：现在 clean_df 已经将标签转换为 int 并过滤了无效行。
train_df = clean_df(train_df)
validation_df = clean_df(validation_df)
test_df = clean_df(test_df)

# ************************************************************
# 移除冗余的标签映射代码，因为标签已经是 0, 1, 2
# ************************************************************
print(f"清洗后训练集有效样本数: {len(train_df)}")

# 转换为 Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

# 删除由 pandas 索引转换产生的冗余列
print("--- 正在处理 Dataset 中的冗余列 ---")
if '__index_level_0__' in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns('__index_level_0__')
if '__index_level_0__' in validation_dataset.column_names:
    validation_dataset = validation_dataset.remove_columns('__index_level_0__')
if '__index_level_0__' in test_dataset.column_names:
    test_dataset = test_dataset.remove_columns('__index_level_0__')

# 创建 DatasetDict
snli_datasets = DatasetDict({
    "train": train_dataset,
    "validation": validation_dataset,
    "test": test_dataset
})

print("数据集处理完毕，开始Bert模型训练")


# -------------------------------------------------------------------------------------
# 从这里开始是针对 RoBERTa 和 XLNet 的独立训练代码
# -------------------------------------------------------------------------------------

# --- 4. XLNet 训练流程 ---

### XLNet 训练代码 (基于 xlnet-base-cased)
MODEL_NAME_XLNET = "xlnet-base-cased"
OUTPUT_DIR_XLNET = "./results/nli_xlnet"

# 初始化分词器
print(f"\n--- 正在初始化 XLNet 分词器 ({MODEL_NAME_XLNET}) ---")
tokenizer_xlnet = AutoTokenizer.from_pretrained(MODEL_NAME_XLNET)

# 应用预处理
tokenized_train_xlnet = train_dataset.map(
    lambda x: preprocess_function(x, tokenizer_xlnet),
    batched=True,
    desc="Tokenizing XLNet Train"
)
tokenized_val_xlnet = validation_dataset.map(
    lambda x: preprocess_function(x, tokenizer_xlnet),
    batched=True,
    desc="Tokenizing XLNet Validation"
)
tokenized_test_xlnet = test_dataset.map(
    lambda x: preprocess_function(x, tokenizer_xlnet),
    batched=True,
    desc="Tokenizing XLNet Test"
)

# 初始化模型
model_xlnet = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_XLNET,
    num_labels=num_labels
)

# 定义训练参数
# 修复后的 TrainingArguments 代码
training_args_xlnet = TrainingArguments(
    output_dir=OUTPUT_DIR_XLNET,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    logging_dir='./logs/xlnet',
    logging_steps=500,
    eval_strategy="epoch",       # <--- 修复: 将 'evaluation_strategy' 改为 'eval_strategy'
    save_strategy="epoch",       # <--- 正确
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
print("\n--- XLNet 在测试集上进行评估 ---")
metrics_xlnet = trainer_xlnet.evaluate(tokenized_test_xlnet)
print(f"XLNet 评估结果: {metrics_xlnet}")