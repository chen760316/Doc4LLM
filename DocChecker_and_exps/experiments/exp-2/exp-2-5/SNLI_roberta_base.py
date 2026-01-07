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
TEMP_LOG_DIR = "./SNLI_RoBERTa_temp_logs"
REPORT_DIR = "./report_SNLI_RoBERTa"
FINAL_REPORT_TXT = os.path.join(REPORT_DIR, "SNLI_RoBERTa_report.txt")
FINAL_REPORT_JSON = os.path.join(REPORT_DIR, "SNLI_metrics_roberta.json")

# 🎯 1. 指标计算函数 (四项指标)
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        logits = preds[0]
    else:
        logits = preds
        
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
    }

# 🎯 2. 数据处理函数
def preprocess_function(examples, tokenizer):
    # RoBERTa 的输入处理
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

def map_labels(examples):
    examples['labels'] = [int(float(lbl)) for lbl in examples['label']]
    return {'labels': examples['labels']}

# --- 3. 主程序 ---
if __name__ == '__main__':
    LIMIT = 900
    # 🎯 替换模型为 roberta-base
    MODEL_CHECKPOINT = "./roberta-base-local"
    
    # 目录准备
    for d in [TEMP_LOG_DIR, REPORT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    data_files = {
        "train": "snli_train.csv",
        "validation": "snli_validation.csv",
        "test": "snli_test.csv"
    }
    
    print(f"✅ 正在加载 RoBERTa 分词器: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    try:
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"❌ 加载 CSV 失败: {e}")
        sys.exit(1)

    # 清洗非法标签
    print("--- 正在清洗数据 ---")
    valid_labels = ["0", "1", "2", 0, 1, 2, "0.0", "1.0", "2.0"]
    raw_datasets = raw_datasets.filter(lambda x: x['label'] in valid_labels and x['premise'] is not None and x['hypothesis'] is not None)

    # 采样
    limited_datasets = DatasetDict({
        split: raw_datasets[split].shuffle(seed=42).select(range(min(LIMIT, len(raw_datasets[split]))))
        for split in raw_datasets
    })

    print("--- 正在进行 RoBERTa 预处理 ---")
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

    # 🎯 加载 RoBERTa 模型
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=3)

    # 训练配置
    training_args = TrainingArguments(
        output_dir=TEMP_LOG_DIR,
        learning_rate=1e-5,            # RoBERTa 通常对较小的学习率更稳定
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",            # 绝不保存中间文件
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

    print("\n--- 🚀 开始 RoBERTa-base 测评训练 ---")
    trainer.train()

    print("\n--- 📊 最终评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # --- 4. 报告生成 ---
    report_content = f"""
--- SNLI (RoBERTa-base) 测评报告 ---
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

    # 清理无用日志
    if os.path.exists(TEMP_LOG_DIR):
        shutil.rmtree(TEMP_LOG_DIR)

    print(f"\n✅ 任务完成！结果目录: {REPORT_DIR}")