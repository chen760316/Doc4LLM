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
    BartForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EvalPrediction
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 环境配置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 0. 路径定义 ---
# 临时工作目录，训练结束后会被删除
TEMP_WORK_DIR = "./ContractNLI_BART_TEMP_DIR" 
# 结果保存目录
REPORT_DIR = "./report_ContractNLI_BART"
FINAL_REPORT_TXT = os.path.join(REPORT_DIR, "ContractNLI_BART_report.txt")
FINAL_REPORT_JSON = os.path.join(REPORT_DIR, "ContractNLI_BART_metrics.json")

# 🎯 1. 自定义 BART 模型类
class BartForContractNLI(torch.nn.Module):
    def __init__(self, model_checkpoint, num_labels=3):
        super().__init__()
        self.bart = BartForSequenceClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels
        )
        self.config = self.bart.config
        self.bart.tie_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return SequenceClassifierOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=None,
            attentions=None,
        )

# 🎯 2. 指标计算函数
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
    }

# 🎯 3. 数据处理
def preprocess_function(examples, tokenizer):
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

# --- 4. 主程序 ---
if __name__ == '__main__':
    LIMIT = 900
    MODEL_CHECKPOINT = "./bart-base" 
    
    # 彻底清理历史遗留目录
    for d in [TEMP_WORK_DIR, REPORT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    data_files = {
        "train": "contractnli_a_train.csv",
        "validation": "contractnli_a_validation.csv",
        "test": "contractnli_a_test.csv"
    }
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    raw_datasets = load_dataset("csv", data_files=data_files)

    # 采样与预处理
    limited_datasets = DatasetDict({
        split: raw_datasets[split].shuffle(seed=42).select(range(min(LIMIT, len(raw_datasets[split]))))
        for split in raw_datasets
    })

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

    model = BartForContractNLI(MODEL_CHECKPOINT, num_labels=3)

    # 训练配置：最小化磁盘占用
    training_args = TrainingArguments(
        output_dir=TEMP_WORK_DIR,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,           # 训练期间只保留一个 checkpoint
        load_best_model_at_end=True,  # 寻找最优模型
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_safetensors=False        # 解决权重共享问题
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

    print("\n--- 📊 测试集评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # --- 5. 仅保存评估结果 ---
    report_content = f"""
--- ContractNLI (BART) 最终评估报告 ---
Accuracy:  {results.get('eval_accuracy', 0.0):.4f}
Precision: {results.get('eval_precision', 0.0):.4f}
Recall:    {results.get('eval_recall', 0.0):.4f}
F1 Score:  {results.get('eval_f1', 0.0):.4f}
---------------------------
"""
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report_content.strip())
    
    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # --- 6. 核心清理逻辑 ---
    print("\n--- 🧹 清理中间 Checkpoints ---")
    if os.path.exists(TEMP_WORK_DIR):
        shutil.rmtree(TEMP_WORK_DIR)
        print(f"已删除临时目录: {TEMP_WORK_DIR}")

    print(f"\n✅ 任务完成！仅保留结果文件在: {REPORT_DIR}")