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
TEMP_LOG_DIR = "./SNLI_BERT_temp_logs"
REPORT_DIR = "./report_SNLI_BERT"
FINAL_REPORT_TXT = os.path.join(REPORT_DIR, "SNLI_BERT_report.txt")
FINAL_REPORT_JSON = os.path.join(REPORT_DIR, "SNLI_metrics.json")

# 🎯 1. 增强后的指标计算函数
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds = eval_pred.predictions
    # 兼容处理：某些模型可能返回 tuple
    if isinstance(preds, tuple):
        logits = preds[0]
    else:
        logits = preds
        
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    # 计算四项核心指标
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
    }

# 🎯 2. 数据处理函数
def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

def map_labels(examples):
    # 确保标签严格为整数
    examples['labels'] = [int(float(lbl)) for lbl in examples['label']]
    return {'labels': examples['labels']}

# --- 3. 主程序 ---
if __name__ == '__main__':
    LIMIT = 900
    MODEL_CHECKPOINT = "bert-base-uncased" 
    
    # 目录准备
    for d in [TEMP_LOG_DIR, REPORT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 文件定义
    data_files = {
        "train": "snli_train.csv",
        "validation": "snli_validation.csv",
        "test": "snli_test.csv"
    }
    
    print(f"✅ 正在加载分词器: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    try:
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"❌ 加载 CSV 失败: {e}")
        sys.exit(1)

    # 关键步骤：过滤非法标签，防止 CUDA 越界
    print("--- 正在清洗数据（剔除非法标签） ---")
    valid_labels = ["0", "1", "2", 0, 1, 2, "0.0", "1.0", "2.0"]
    raw_datasets = raw_datasets.filter(lambda x: x['label'] in valid_labels and x['premise'] is not None and x['hypothesis'] is not None)

    # 采样
    limited_datasets = DatasetDict({
        split: raw_datasets[split].shuffle(seed=42).select(range(min(LIMIT, len(raw_datasets[split]))))
        for split in raw_datasets
    })

    print("--- 正在分词预处理 ---")
    tokenized_datasets = limited_datasets.map(
        preprocess_function, 
        batched=True, 
        fn_kwargs={"tokenizer": tokenizer}
    ).map(
        map_labels, 
        batched=True
    )

    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "label"])
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=3)

    # 训练配置
    training_args = TrainingArguments(
        output_dir=TEMP_LOG_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=50,
        remove_unused_columns=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n--- 🚀 开始测评训练 ---")
    try:
        trainer.train()
    except RuntimeError as e:
        print(f"❌ 运行报错: {e}")
        sys.exit(1)

    print("\n--- 📊 最终评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # --- 4. 报告保存 (包含四项指标) ---
    report_content = f"""
--- SNLI (BERT-base) 测评报告 ---
Accuracy:  {results.get('eval_accuracy', 0.0):.4f}
Precision: {results.get('eval_precision', 0.0):.4f} (Weighted)
Recall:    {results.get('eval_recall', 0.0):.4f} (Weighted)
F1 Score:  {results.get('eval_f1', 0.0):.4f} (Weighted)
---------------------------
"""
    print(report_content)
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report_content.strip())
    
    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # 清理日志
    if os.path.exists(TEMP_LOG_DIR):
        shutil.rmtree(TEMP_LOG_DIR)

    print(f"\n✅ 任务完成！结果目录: {REPORT_DIR}")