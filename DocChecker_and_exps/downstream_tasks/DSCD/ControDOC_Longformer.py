# DSCD_classify_nested.py
"""
使用的数据集：ContraDOC
使用的基准："allenai/longformer-base-4096"
"""
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
import json
import os
from typing import Dict, Any, Iterator, Tuple


# --- 1. 定制数据加载函数 ---

def generate_examples(filepaths: Dict[str, str]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """
    加载并展平 ContraDOC 的嵌套 JSON 文件。

    Args:
        filepaths: 包含文件路径的字典，例如 {"train": "path/to/ContraDoc.json"}

    Yields:
        (index, data_dict) 格式的元组，供 datasets.load_dataset 使用。
    """
    if "train" not in filepaths:
        raise ValueError("File paths must contain the 'train' key pointing to ContraDoc.json.")

    file_path = filepaths["train"]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    all_examples = []

    # 处理 "pos" (有矛盾, label=1)
    if "pos" in data:
        for unique_id, doc_data in data["pos"].items():
            all_examples.append({
                "document": doc_data["text"],
                "label": 1,  # 1 代表有矛盾 (Contradictory)
                "unique_id": unique_id
            })

    # 处理 "neg" (无矛盾, label=0)
    if "neg" in data:
        for unique_id, doc_data in data["neg"].items():
            all_examples.append({
                "document": doc_data["text"],
                "label": 0,  # 0 代表无矛盾 (Non-Contradictory)
                "unique_id": unique_id
            })

    # 按照索引和字典格式返回数据，供 datasets 内部处理
    for i, example in enumerate(all_examples):
        yield i, example


# --- 2. 预处理函数 ---

def preprocess_function(examples, tokenizer):
    """
    对 ContraDOC 数据集进行分词。
    """
    return tokenizer(
        examples["document"],
        truncation=True,
        max_length=4096
    )


def compute_metrics(eval_pred):
    """计算评估指标（准确率）"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


# --- 3. 核心执行块 ---

if __name__ == '__main__':
    # 限制的记录数（用于快速测试）
    LIMIT = 1000  # 设置一个较小的限制以确保代码能运行

    # 您的文件路径
    data_files = {
        "train": "../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"
    }

    # 1. 定义类别数
    num_labels = 2

    # 2. 选择和加载模型/分词器
    MODEL_CHECKPOINT = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 3. 手动加载 ContraDOC JSON 文件并展平
    print("🌍 正在手动加载 ContraDOC JSON 文件并展平结构...")
    file_path = data_files["train"]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: ContraDOC file not found at {file_path}")
        exit()

    # 展平数据结构
    flattened_data = []

    # 0. 处理 "pos" (有矛盾, label=1)
    if "pos" in data:
        for unique_id, doc_data in data["pos"].items():
            flattened_data.append({
                "document": doc_data["text"],
                "label": 1,
                "unique_id": unique_id
            })

    # 1. 处理 "neg" (无矛盾, label=0)
    if "neg" in data:
        for unique_id, doc_data in data["neg"].items():
            flattened_data.append({
                "document": doc_data["text"],
                "label": 0,
                "unique_id": unique_id
            })

    # 转换为字典列表，然后使用 Dataset.from_list 创建 Dataset 对象
    raw_dataset = Dataset.from_list(flattened_data)

    # 4. 划分数据集为 80% 训练, 10% 验证, 10% 测试
    print(f"⚠️ 原始数据有 {len(raw_dataset)} 条，正在进行训练/验证/测试集划分 (80/10/10)...")

    # 第一次划分：分离出 20%
    train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_test_split['train']
    test_dataset = train_test_split['test']

    # 第二次划分：从剩余的 80% 中分离出 10% 作为验证集 (0.1 / 0.8 = 0.125)
    train_val_split = train_val.train_test_split(test_size=0.125, seed=42)  # 0.125 确保验证集占总体的 10%
    train_dataset = train_val_split['train']
    validation_dataset = train_val_split['test']

    # 将划分结果合并到 DatasetDict
    limited_datasets = DatasetDict({
        "train": train_dataset.select(range(min(LIMIT, len(train_dataset)))) if LIMIT else train_dataset,
        "validation": validation_dataset.select(range(min(LIMIT, len(validation_dataset)))) if LIMIT else validation_dataset,
        "test": test_dataset.select(range(min(LIMIT, len(test_dataset)))) if LIMIT else test_dataset,
    })

    # 5. 文本预处理 (分词)
    print("⏳ 正在进行文本分词...")
    tokenized_datasets = limited_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 6. 设置数据格式
    # 标签列名已经是 'label'，无需 rename_column，但需确保 Trainer 知道它是标签
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    print(f"✅ 数据集加载、分词、划分完成 (类别数: {num_labels})。")
    print(tokenized_datasets)

    # 7. 模型加载
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)

    # 8. 定义训练参数
    training_args = TrainingArguments(
        output_dir="./dscd_results",
        learning_rate=2e-5,
        per_device_train_batch_size=2,  # 保持较小 batch size
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./dscd_logs',
        logging_steps=50,
        load_best_model_at_end=True,
    )

    # 9. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 10. 开始训练
    print("\n--- 🚀 开始 DSCD 模型训练 ---")
    trainer.train()

    # 11. 在测试集上评估
    print("\n--- 📊 在测试集上评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(results)