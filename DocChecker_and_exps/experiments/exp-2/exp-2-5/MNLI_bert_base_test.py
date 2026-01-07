# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import shutil
import numpy as np
from typing import Dict
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EvalPrediction
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 0. 路径定义 ---
TEMP_LOG_DIR = "./MNLI_BERT_temp_logs"
REPORT_DIR = "./report_MNLI_Final"
FINAL_REPORT_TXT = os.path.join(REPORT_DIR, "MNLI_report.txt")
FINAL_REPORT_JSON = os.path.join(REPORT_DIR, "MNLI_metrics.json")

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds = eval_pred.predictions
    logits = preds[0] if isinstance(preds, tuple) else preds
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
    }

# 🎯 2. 针对 text1 和 text2 的预处理
def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text1"],
        examples["text2"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

def map_labels(example):
    # 直接取 label 列，并确保是整数
    try:
        l = int(float(example['label']))
    except:
        l = -1
    return {'labels': l}

if __name__ == '__main__':
    LIMIT_TRAIN = 2000 
    LIMIT_VAL_TEST = 500 # 每个集 500 条
    MODEL_CHECKPOINT = "bert-base-uncased" # 或你本地路径
    
    for d in [TEMP_LOG_DIR, REPORT_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 加载数据
    data_files = {"train": "mnli_train.csv", "validation": "mnli_validation.csv"}
    print("🚀 正在加载 CSV 数据...")
    raw_datasets = load_dataset("csv", data_files=data_files)

    print("--- 正在根据你的格式清洗数据 ---")
    # 映射标准 labels 列并过滤
    processed = raw_datasets.map(map_labels)
    valid_datasets = processed.filter(
        lambda x: x['labels'] in [0, 1, 2] and x['text1'] is not None and x['text2'] is not None
    )

    # 检查是否还有数据
    train_count = len(valid_datasets["train"])
    val_count = len(valid_datasets["validation"])
    print(f"✅ 清洗完成。训练集可用: {train_count}, 验证集可用: {val_count}")

    if train_count == 0 or val_count == 0:
        print("❌ 错误：过滤后数据集为空！请检查 CSV 编码或列名是否完全匹配 'text1', 'text2', 'label'")
        sys.exit(1)

    # 🎯 3. 划分验证集和测试集 (从原始 validation 划分)
    print("--- 正在切分独立测试集 ---")
    val_data = valid_datasets["validation"].shuffle(seed=42)
    # 取前 1000 条，500给验证，500给测试
    val_test_subset = val_data.select(range(min(LIMIT_VAL_TEST * 2, len(val_data))))
    split_result = val_test_subset.train_test_split(test_size=0.5, seed=42)

    final_dict = DatasetDict({
        "train": valid_datasets["train"].shuffle(seed=42).select(range(min(LIMIT_TRAIN, train_count))),
        "validation": split_result["train"],
        "test": split_result["test"]
    })

    print(f"📊 最终规模 -> 训练: {len(final_dict['train'])}, 验证: {len(final_dict['validation'])}, 测试: {len(final_dict['test'])}")

    # 4. 分词
    print("⏳ 正在进行分词预处理...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenized_datasets = final_dict.map(
        preprocess_function, 
        batched=True, 
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 设置 Torch 格式
    keep_cols = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    tokenized_datasets.set_format(type="torch", columns=[c for c in keep_cols if c in tokenized_datasets['train'].column_names])

    # 5. 训练配置
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=3)
    
    training_args = TrainingArguments(
        output_dir=TEMP_LOG_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n--- 🚀 开始训练 ---")
    trainer.train()

    print("\n--- 📊 最终独立测试集评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # 报告打印与保存
    report = f"""
--- MNLI 测评报告 (独立测试集) ---
模型: {MODEL_CHECKPOINT}
Accuracy:  {results.get('eval_accuracy', 0.0):.4f}
Precision: {results.get('eval_precision', 0.0):.4f}
Recall:    {results.get('eval_recall', 0.0):.4f}
F1 Score:  {results.get('eval_f1', 0.0):.4f}
---------------------------
"""
    print(report)
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f: f.write(report.strip())
    
    if os.path.exists(TEMP_LOG_DIR): shutil.rmtree(TEMP_LOG_DIR)
    print(f"✅ 任务完成！结果目录: {REPORT_DIR}")