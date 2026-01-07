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
TEMP_LOG_DIR = "./MNLI_RoBERTa_temp_logs"
REPORT_DIR = "./report_MNLI_RoBERTa"
FINAL_REPORT_TXT = os.path.join(REPORT_DIR, "MNLI_RoBERTa_report.txt")
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

def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text1"],
        examples["text2"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

def map_labels(example):
    try:
        l = int(float(example['label']))
    except:
        l = -1
    return {'labels': l}

if __name__ == '__main__':
    # --- 🎯 替换为本地模型路径 ---
    MODEL_CHECKPOINT = "./roberta-base-local"  # 请确保该目录下有 config.json 和 pytorch_model.bin
    LIMIT_TRAIN = 2000 
    LIMIT_VAL_TEST = 500 
    
    for d in [TEMP_LOG_DIR, REPORT_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 1. 加载数据
    data_files = {"train": "mnli_train.csv", "validation": "mnli_validation.csv"}
    print(f"🚀 正在加载 CSV 数据...")
    raw_datasets = load_dataset("csv", data_files=data_files)

    # 2. 清洗与切分
    processed = raw_datasets.map(map_labels)
    valid_datasets = processed.filter(
        lambda x: x['labels'] in [0, 1, 2] and x['text1'] is not None and x['text2'] is not None
    )

    val_data = valid_datasets["validation"].shuffle(seed=42)
    val_test_subset = val_data.select(range(min(LIMIT_VAL_TEST * 2, len(val_data))))
    split_result = val_test_subset.train_test_split(test_size=0.5, seed=42)

    final_dict = DatasetDict({
        "train": valid_datasets["train"].shuffle(seed=42).select(range(min(LIMIT_TRAIN, len(valid_datasets["train"])))),
        "validation": split_result["train"],
        "test": split_result["test"]
    })

    # 3. 分词预处理
    print(f"⏳ 正在加载本地分词器: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    tokenized_datasets = final_dict.map(
        preprocess_function, 
        batched=True, 
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 🎯 RoBERTa 关键点：设置格式时排除 token_type_ids（RoBERTa 不需要它）
    keep_cols = ['input_ids', 'attention_mask', 'labels']
    tokenized_datasets.set_format(type="torch", columns=keep_cols)

    # 4. 加载本地模型
    print(f"📦 正在加载本地模型: {MODEL_CHECKPOINT}")
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

    print("\n--- 🚀 开始训练 (RoBERTa) ---")
    trainer.train()

    print("\n--- 📊 最终独立测试集评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # 5. 报告保存
    report = f"""
--- MNLI 测评报告 (RoBERTa-base 本地) ---
本地路径: {MODEL_CHECKPOINT}
Accuracy:  {results.get('eval_accuracy', 0.0):.4f}
Precision: {results.get('eval_precision', 0.0):.4f}
Recall:    {results.get('eval_recall', 0.0):.4f}
F1 Score:  {results.get('eval_f1', 0.0):.4f}
---------------------------
"""
    print(report)
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f: f.write(report.strip())
    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f: json.dump(results, f, indent=4)
    
    if os.path.exists(TEMP_LOG_DIR): shutil.rmtree(TEMP_LOG_DIR)
    print(f"✅ 任务完成！结果目录: {REPORT_DIR}")