# -*- coding: utf-8 -*-
# DocNLI_classify_ternary.py
"""
使用的数据集：ContractNLI (三分类 NLI)
使用的基准："allenai/longformer-base-4096"
修复重点：解决 map_labels 参数冗余导致的 TypeError，并确保标签转换为整数。
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
# 临时工作目录，用于存放训练过程中的 Checkpoint
TEMP_WORK_DIR = "./ContractNLI_results_TEMP"
LOGGING_DIR = os.path.join(TEMP_WORK_DIR, 'logs')

# 最终报告存储目录
REPORT_DIR = "./report"
FINAL_REPORT_TXT = os.path.join(REPORT_DIR, "ContractNLI_classify_longformer_report.txt")
FINAL_REPORT_JSON = os.path.join(REPORT_DIR, "ContractNLI_ternary_metrics.json")

# --- 1. 核心辅助函数 ---

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """计算四项核心指标 (Weighted)"""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
    }

def preprocess_function(examples, tokenizer):
    """分词处理 (针对 Longformer 4096)"""
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=4096,
        padding="max_length"  # 确保固定长度以稳定显存
    )

def map_labels(examples):
    """
    确保标签列为整数类型。
    不再需要额外的 label_to_id 字典，直接转换数据中已有的数字标签。
    """
    try:
        # 兼容处理：支持字符串、浮点数转整数 (如 "1" 或 "1.0" -> 1)
        examples['labels'] = [int(float(lbl)) for lbl in examples['label']]
    except Exception as e:
        sample_val = examples['label'][:3] if 'label' in examples else "None"
        raise ValueError(f"标签转换失败。读取到的值为: {sample_val}。错误原因: {e}")
    return {'labels': examples['labels']}

# --- 2. 核心执行逻辑 ---

if __name__ == '__main__':
    # 配置参数
    LIMIT = 900
    MODEL_CHECKPOINT = "allenai/longformer-base-4096"
    
    # 目录准备
    os.makedirs(REPORT_DIR, exist_ok=True)
    if os.path.exists(TEMP_WORK_DIR):
        print(f"🗑️ 清理旧的临时目录: {TEMP_WORK_DIR}")
        shutil.rmtree(TEMP_WORK_DIR)
    os.makedirs(TEMP_WORK_DIR, exist_ok=True)

    # 数据加载
    data_files = {
        "train": "contractnli_a_train.csv",
        "validation": "contractnli_a_validation.csv",
        "test": "contractnli_a_test.csv"
    }
    
    print(f"✅ 加载模型: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    try:
        # 设置镜像源以防下载失败
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"❌ 加载 CSV 失败: {e}")
        sys.exit(1)

    # 限制样本量并预处理
    print("--- 正在进行预处理 (分词与标签映射) ---")
    limited_datasets = DatasetDict({
        split: raw_datasets[split].select(range(min(LIMIT, len(raw_datasets[split]))))
        for split in raw_datasets
    })

    # 第一次 map: 分词
    tokenized_datasets = limited_datasets.map(
        preprocess_function, 
        batched=True, 
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 第二次 map: 标签映射 (已修复 TypeError: 不再传递 label_to_id)
    tokenized_datasets = tokenized_datasets.map(
        map_labels, 
        batched=True, 
        num_proc=4
    )

    # 清理列名并设置格式
    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "label"])
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    # 模型加载 (三分类)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=3)

    # --- 3. 训练配置 (显存优化关键点) ---
    training_args = TrainingArguments(
        output_dir=TEMP_WORK_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=1,     # 4096长度下，设为1防止显存溢出
        per_device_eval_batch_size=1,      # 评估也设为1
        gradient_accumulation_steps=8,     # 累加8步，实际等效 batch_size=8
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),    # 启用混合精度
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

    # 开始训练
    print("\n--- 🚀 开始 ContractNLI 训练 (Longformer 4096) ---")
    trainer.train()

    # 测试集评估
    print("\n--- 📊 测试集评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # --- 4. 报告保存逻辑 ---
    accuracy = results.get('eval_accuracy', 0.0)
    precision = results.get('eval_precision', 0.0)
    recall = results.get('eval_recall', 0.0)
    f1 = results.get('eval_f1', 0.0)

    report_content = f"""
--- ContractNLI 三元分类评估报告 ---
模型: {MODEL_CHECKPOINT}
序列长度: 4096
测试样本数: {len(tokenized_datasets["test"])}
---------------------------
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f} (Weighted)
Recall:    {recall:.4f} (Weighted)
F1 Score:  {f1:.4f} (Weighted)
---------------------------
"""
    # 保存文本报告
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report_content.strip())
    
    # 保存详细 JSON
    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # 保存最终微调模型
    final_model_path = os.path.join(REPORT_DIR, "final_contractnli_model")
    trainer.save_model(final_model_path)

    # --- 5. 彻底清理临时文件 ---
    if os.path.exists(TEMP_WORK_DIR):
        print(f"\n🗑️ 正在清理临时空间 ({TEMP_WORK_DIR})...")
        shutil.rmtree(TEMP_WORK_DIR)
        print("✅ 临时 Checkpoints 已删除。")

    print(f"\n✅ 任务全部完成！")
    print(f"📊 报告路径: {FINAL_REPORT_TXT}")
    print(f"📦 模型路径: {final_model_path}")