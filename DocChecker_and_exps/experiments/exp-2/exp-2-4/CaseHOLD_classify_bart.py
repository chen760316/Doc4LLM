# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import numpy as np
import evaluate
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

# 1. 设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 0. 结果保存路径定义 ---
OUTPUT_DIR = "./CaseHOLD_MCQA_BART_results"
LOGGING_DIR = './CaseHOLD_MCQA_BART_logs'
FINAL_REPORT_TXT = os.path.join(OUTPUT_DIR, "report/CaseHOLD_classify_bart_report.txt")
FINAL_REPORT_JSON = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")

# 🎯 1. 自定义 BART 多项选择模型类
class BartMultipleChoiceModel(torch.nn.Module):
    def __init__(self, model_checkpoint):
        super().__init__()
        self.bart = BartForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)
        self.config = self.bart.config
        # 💡 确保权重绑定逻辑在自定义包装器中也生效
        self.bart.tie_weights()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        num_choices = input_ids.shape[1]
        
        # 展平输入 (batch_size * num_choices, seq_len)
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        
        outputs = self.bart(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        logits = outputs.logits # (batch_size * num_choices, 1)
        
        # 塑回 (batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            
        # 💡 修复点：只返回 Trainer 必需的 loss 和 logits
        # 这样可以避免 AttributeError，因为 SequenceClassifierOutput 允许这些字段为 None
        return SequenceClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=None,
            attentions=None,
        )

# 🎯 2. 核心修复：指标计算函数
def compute_metrics_mcqa(eval_pred: EvalPrediction) -> Dict[str, float]:
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

# 🎯 3. 数据预处理
def preprocess_function_mcqa(examples, tokenizer):
    context_name = "citing_prompt"
    num_options = 5

    first_sentences = [[context] * num_options for context in examples[context_name]]
    second_sentences = []
    for i in range(len(examples[context_name])):
        options = [examples[f"holding_{j}"][i] for j in range(num_options)]
        second_sentences.append(options)

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    return {
        **{k: [v[i:i + num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()},
        "labels": examples["label"],
    }

# --- 4. 主程序 ---
if __name__ == '__main__':
    LIMIT = 500
    os.makedirs(os.path.dirname(FINAL_REPORT_TXT), exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    data_files = {
        "train": "casehold_train.csv",
        "validation": "casehold_validation.csv",
        "test": "casehold_test.csv"
    }

    MODEL_CHECKPOINT = "./bart-base"
    print(f"✅ 任务: CaseHOLD MCQA | 模型: {MODEL_CHECKPOINT} (Custom Wrapper)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    try:
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载数据: {e}", file=sys.stderr)
        sys.exit(1)

    limited_datasets = DatasetDict({
        split: raw_datasets[split].select(range(min(LIMIT, len(raw_datasets[split]))))
        for split in raw_datasets
    })

    tokenized_datasets = limited_datasets.map(
        preprocess_function_mcqa,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    model = BartMultipleChoiceModel(MODEL_CHECKPOINT)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        logging_dir=LOGGING_DIR,
        # 💡 核心修复：禁用 safetensors 以允许共享权重的保存
        save_safetensors=False 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mcqa,
    )

    print("\n--- 🚀 开始训练 ---")
    trainer.train()

    print("\n--- 📊 测试集评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # 提取指标
    accuracy = results.get('eval_accuracy', 0.0)
    precision = results.get('eval_precision', 0.0)
    recall = results.get('eval_recall', 0.0)
    f1 = results.get('eval_f1', 0.0)

    print(f"\n最终测试集评估结果:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 保存报告
    final_text_content = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}"
    with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(final_text_content)

    with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "bart_mcqa_model.pt"))
    print(f"\n✅ 任务完成！")