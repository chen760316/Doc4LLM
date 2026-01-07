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
TEMP_LOG_DIR = "./ContractNLI_DistilBERT_temp_logs"
REPORT_DIR = "./report_ContractNLI_DistilBERT"
FINAL_REPORT_TXT = os.path.join(REPORT_DIR, "ContractNLI_DistilBERT_report.txt")
FINAL_REPORT_JSON = os.path.join(REPORT_DIR, "ContractNLI_DistilBERT_metrics.json")

# 🎯 1. 修复后的指标计算函数
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    # 这里的 predictions 可能是 (logits,) 或者 (logits, hidden_states)
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

# 🎯 2. 数据处理
def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

def map_labels(examples):
    # 确保标签是长整型
    examples['labels'] = [int(float(lbl)) for lbl in examples['label']]
    return {'labels': examples['labels']}

# --- 3. 主程序 ---
if __name__ == '__main__':
    LIMIT = 900
    MODEL_CHECKPOINT = "distilbert-base-uncased" 
    
    # 彻底清理历史目录释放空间
    for d in [TEMP_LOG_DIR, REPORT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    data_files = {
        "train": "contractnli_a_train.csv",
        "validation": "contractnli_a_validation.csv",
        "test": "contractnli_a_test.csv"
    }
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    try:
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        sys.exit(1)

    # 随机采样
    limited_datasets = DatasetDict({
        split: raw_datasets[split].shuffle(seed=42).select(range(min(LIMIT, len(raw_datasets[split]))))
        for split in raw_datasets
    })

    print("--- 正在进行分词与预处理 ---")
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

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=3)

    # 训练配置：禁用所有中间文件保存
    training_args = TrainingArguments(
        output_dir=TEMP_LOG_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",            # 不保存 checkpoint
        load_best_model_at_end=False,  # 不在磁盘保存最优模型
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

    print("\n--- 🚀 开始 DistilBERT 训练 (内存模式) ---")
    trainer.train()

    print("\n--- 📊 最终评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # --- 4. 报告生成 ---
    report_content = f"""
--- ContractNLI (DistilBERT) 评估报告 ---
Accuracy:  {results.get('eval_accuracy', 0.0):.4f}
Precision: {results.get('eval_precision', 0.0):.4f}
Recall:    {results.get('eval_recall', 0.0):.4f}
F1 Score:  {results.get('eval_f1', 0.0):.4f}
---------------------------
"""
    print(report_content)
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report_content.strip())
    
    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # 训练后立即清理日志文件夹
    if os.path.exists(TEMP_LOG_DIR):
        shutil.rmtree(TEMP_LOG_DIR)

    print(f"\n✅ 任务完成！结果已存至: {REPORT_DIR}")