# -*- coding: utf-8 -*-
# DocNLI_classify_ternary_deberta.py
"""
使用的数据集：ContractNLI (三分类 NLI)
使用的基准："microsoft/deberta-v3-base"
注意：DeBERTa-v3 最大长度通常为 512，超过部分会被截断。
"""

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EvalPrediction
)
import numpy as np
import torch
import os
import sys
import json
import shutil
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 0. 路径与目录定义 ---
TEMP_WORK_DIR = "./ContractNLI_results_TEMP"
LOGGING_DIR = os.path.join(TEMP_WORK_DIR, 'logs')
REPORT_DIR = "./report"
FINAL_REPORT_TXT = os.path.join(REPORT_DIR, "ContractNLI_classify_deberta_report.txt")
FINAL_REPORT_JSON = os.path.join(REPORT_DIR, "ContractNLI_ternary_metrics_deberta.json")

# --- 1. 核心辅助函数 ---

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    计算多项选择题 (MCQA) 的准确率、精确度、召回率和 F1 分数。
    直接使用 Scikit-learn 库，避免 evaluate 缓存问题。
    使用 'weighted' average 处理多分类任务。
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # 预测结果是 logits 最大值对应的索引 (即所选的选项)
    predictions = np.argmax(logits, axis=-1)

    # ----------------------------------------------------
    # 核心修改：直接调用 sklearn 函数
    # 'weighted' 参数确保适用于多分类和类别不平衡的情况
    # ----------------------------------------------------
    
    # 准确率 (Accuracy)
    accuracy = accuracy_score(labels, predictions)
    
    # 精确度 (Precision) - 多分类 'weighted'
    # zero_division=0 用于防止计算过程中因除零而崩溃
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    
    # 召回率 (Recall) - 多分类 'weighted'
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    
    # F1 Score - 多分类 'weighted'
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    
    # 返回统一的键名
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def preprocess_function(examples, tokenizer):
    """DeBERTa 分词处理 (通常最大长度为 512)"""
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=512, # DeBERTa-v3 默认最大长度
        padding="max_length"
    )

def map_labels(examples):
    try:
        examples['labels'] = [int(float(lbl)) for lbl in examples['label']]
    except Exception as e:
        sample_val = examples['label'][:3] if 'label' in examples else "None"
        raise ValueError(f"标签转换失败: {sample_val}。错误: {e}")
    return {'labels': examples['labels']}

# --- 2. 核心执行逻辑 ---

if __name__ == '__main__':
    LIMIT = 900
    # 🎯 更改为 microsoft/deberta-v3-base
    MODEL_CHECKPOINT = "microsoft/deberta-v3-base"
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    if os.path.exists(TEMP_WORK_DIR):
        shutil.rmtree(TEMP_WORK_DIR)
    os.makedirs(TEMP_WORK_DIR, exist_ok=True)

    data_files = {
        "train": "contractnli_a_train.csv",
        "validation": "contractnli_a_validation.csv",
        "test": "contractnli_a_test.csv"
    }
    
    print(f"✅ 加载模型: {MODEL_CHECKPOINT}")
    # 注意：DeBERTa-v3 使用了 SentencePiece，确保环境已安装相关库
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"❌ 加载 CSV 失败: {e}")
        sys.exit(1)

    # 随机采样 900 条（加入 shuffle 提升代表性）
    print("--- 正在进行随机采样与预处理 ---")
    limited_datasets = DatasetDict({
        split: raw_datasets[split].shuffle(seed=42).select(range(min(LIMIT, len(raw_datasets[split]))))
        for split in raw_datasets
    })

    tokenized_datasets = limited_datasets.map(
        preprocess_function, 
        batched=True, 
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    ).map(
        map_labels, 
        batched=True, 
        num_proc=4
    )

    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "label"])
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=3)

    # --- 3. 训练配置 (DeBERTa-v3 优化) ---
    training_args = TrainingArguments(
        output_dir=TEMP_WORK_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=4,     # 512长度下，Batch Size 可以稍大
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,     # 实际 Batch Size 为 8
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
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

    print("\n--- 🚀 开始 DeBERTa 训练 ---")
    trainer.train()

    print("\n--- 📊 测试集评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # --- 4. 报告保存 ---
    report_content = f"""
--- ContractNLI 三元分类评估报告 ---
模型: {MODEL_CHECKPOINT}
序列长度: 512 (Max)
测试样本数: {len(tokenized_datasets["test"])}
---------------------------
Accuracy:  {results.get('eval_accuracy', 0.0):.4f}
Precision: {results.get('eval_precision', 0.0):.4f} (Weighted)
Recall:    {results.get('eval_recall', 0.0):.4f} (Weighted)
F1 Score:  {results.get('eval_f1', 0.0):.4f} (Weighted)
---------------------------
"""
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report_content.strip())
    
    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    trainer.save_model(os.path.join(REPORT_DIR, "final_deberta_model"))

    if os.path.exists(TEMP_WORK_DIR):
        shutil.rmtree(TEMP_WORK_DIR)

    print(f"\n✅ 任务完成！报告见: {FINAL_REPORT_TXT}")