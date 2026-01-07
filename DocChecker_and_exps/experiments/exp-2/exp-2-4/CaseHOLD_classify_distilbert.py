# -*- coding: utf-8 -*-
# CaseHOLD_MCQA_DistilBERT_classify.py
"""
使用的数据集：CaseHOLD (多项选择问答 MCQA)
使用的基准："distilbert-base-uncased"
功能：计算并保存 Accuracy, Precision, Recall, F1 四个指标 (多分类 weighted)。
"""
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer, EvalPrediction
import numpy as np
import torch
import os
import sys
import json
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 设置镜像源 (AutoDL 环境加速)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 0. 结果保存路径定义 ---
OUTPUT_DIR = "report/CaseHOLD_MCQA_DistilBERT_results"
LOGGING_DIR = 'report/CaseHOLD_MCQA_DistilBERT_logs'
FINAL_REPORT_TXT = os.path.join(OUTPUT_DIR, "report/CaseHOLD_classify_distilbert_report.txt")
FINAL_REPORT_JSON = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")

# 🎯 指标计算函数：直接使用 Scikit-learn
def compute_metrics_mcqa(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def preprocess_function_mcqa(examples, tokenizer):
    context_name = "citing_prompt"
    num_options = 5

    # 1. 构造 Context - Option 对
    contexts = [[examples[context_name][i]] * num_options for i in range(len(examples[context_name]))]
    options = []
    for i in range(num_options):
        options.append(examples[f"holding_{i}"])

    # 2. 展平并分词
    options_transposed = [list(t) for t in zip(*options)]
    contexts_flat = sum(contexts, [])
    options_flat = sum(options_transposed, [])

    tokenized_examples = tokenizer(
        contexts_flat,
        options_flat,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    # 3. 重塑输出张量：(batch_size * 5) -> (batch_size, 5, seq_len)
    return {
        **{k: [v[i:i + num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()},
        "labels": examples["label"],
    }

# ------------------- 核心执行块 -------------------
if __name__ == '__main__':
    LIMIT = 950
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "report"), exist_ok=True)

    data_files = {
        "train": "casehold_train.csv",
        "validation": "casehold_validation.csv",
        "test": "casehold_test.csv"
    }

    # 🎯 修改为 DistilBERT
    MODEL_CHECKPOINT = "distilbert-base-uncased"
    print(f"✅ 模型基准: {MODEL_CHECKPOINT}")

    # 1. 加载分词器 (DistilBERT 不需要 trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 2. 加载数据集
    try:
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载数据: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. 限制数据并预处理
    limited_datasets = DatasetDict({
        split: raw_datasets[split].select(range(min(LIMIT, len(raw_datasets[split]))))
        for split in raw_datasets
    })

    print("--- 正在使用 DistilBERT 格式进行分词 ---")
    tokenized_datasets = limited_datasets.map(
        preprocess_function_mcqa,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    # 4. 模型加载
    try:
        model = AutoModelForMultipleChoice.from_pretrained(MODEL_CHECKPOINT)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载模型权重: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. 定义训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=3e-5,          
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        # 🎯 修复点：将 evaluation_strategy 改为 eval_strategy
        eval_strategy="epoch",  
        save_strategy="no",           
        logging_steps=100,
        report_to="none"
    )

    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mcqa,
    )

    # 7. 开始训练
    print(f"\n--- 🚀 开始训练 (DistilBERT) ---")
    trainer.train()

# 8. 测试集评估
    results = trainer.evaluate(tokenized_datasets["test"])
    
    # 9. 提取四项关键指标 (注意 Trainer 输出的 key 会带有 'eval_' 前缀)
    accuracy = results.get('eval_accuracy', 0.0)
    precision = results.get('eval_precision', 0.0)
    recall = results.get('eval_recall', 0.0)
    f1 = results.get('eval_f1', 0.0)
    
    # 10. 构建报告内容
    final_text_content = f"""
--- CaseHOLD MCQA 评估报告 (DistilBERT) ---
模型基准: {MODEL_CHECKPOINT}
测试样本数: {len(tokenized_datasets["test"])}
---------------------------------------
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f} (Weighted)
Recall:    {recall:.4f} (Weighted)
F1 Score:  {f1:.4f} (Weighted)
---------------------------------------
"""
    
    # 保存为 TXT 报告
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(final_text_content.strip())
    
    # 保存为详细 JSON (包含所有原始数据)
    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(final_text_content) # 在终端直观打印一次
    print(f"✅ 任务完成。完整报告已保存至: {FINAL_REPORT_TXT}")