# CaseHOLD_MCQA_classify.py
"""
使用的数据集：CaseHOLD (多项选择问答 MCQA)
使用的基准："allenai/longformer-base-4096"
"""
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForMultipleChoice, 
    TrainingArguments, 
    Trainer, 
    EvalPrediction,
    # 移除 DataCollatorForMultipleChoice 和 DataCollatorWithPadding
)
import numpy as np
import evaluate
import torch
import os
import sys
import json
import shutil # 用于文件和目录操作
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 0. 结果保存路径定义 ---
# OUTPUT_DIR 作为 Trainer 的临时工作目录
TEMP_WORK_DIR = "./CaseHOLD_MCQA_results_TEMP" 
LOGGING_DIR = os.path.join(TEMP_WORK_DIR, 'logs')

# 最终报告将保存到的目标目录（相对路径）
FINAL_REPORT_TARGET_DIR = "./report" 
# 最终报告文件的完整路径（在程序结束时创建）
FINAL_REPORT_TXT = os.path.join(FINAL_REPORT_TARGET_DIR, "CaseHOLD_classify_longformer_report.txt") 
FINAL_REPORT_JSON = os.path.join(FINAL_REPORT_TARGET_DIR, "evaluation_metrics.json")


# 🎯 指标计算函数（保持不变）
def compute_metrics_mcqa(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    计算多项选择题 (MCQA) 的准确率、精确度、召回率和 F1 分数。
    """
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
    """
    针对 CaseHOLD 多项选择问答任务的预处理函数。
    """
    context_name = "citing_prompt"
    num_options = 5

    # 1. 复制 context
    contexts = [[examples[context_name][i]] * num_options for i in range(len(examples[context_name]))]

    # 2. 收集所有选项 (代码保持不变)
    options = []
    for i in range(num_options):
        option_col_name = f"holding_{i}"
        try:
            option_column = examples[option_col_name]
        except KeyError:
            print(f"FATAL ERROR: 缺少 CaseHOLD 选项列。请检查您的 CSV 中是否有 '{option_col_name}' 等列名。", file=sys.stderr)
            raise

        options.append(option_column)

    # 3. 展平选项 (代码保持不变)
    options_transposed_of_tuples = zip(*options)
    options_transposed = [list(t) for t in options_transposed_of_tuples]

    # 4. 展平上下文和选项 (代码保持不变)
    contexts = sum(contexts, [])
    options_flat = sum(options_transposed, [])

    # 5. 分词
    tokenized_examples = tokenizer(
        contexts,
        options_flat,
        truncation=True,
        max_length=2048, # 保持 2048 的序列长度
        padding="max_length" # 必须启用填充
    )

    # 6. 重塑输出张量
    return {
        **{k: [v[i:i + num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()},
        "labels": examples["label"],
    }


# ------------------- 核心执行块 -------------------
if __name__ == '__main__':
    LIMIT = 100
    
    # --- 目录准备和清理 ---
    FINAL_REPORT_TARGET_DIR = "./report"
    FINAL_REPORT_TXT = os.path.join(FINAL_REPORT_TARGET_DIR, "CaseHOLD_classify_longformer_report.txt")
    FINAL_REPORT_JSON = os.path.join(FINAL_REPORT_TARGET_DIR, "evaluation_metrics.json")
    
    os.makedirs(FINAL_REPORT_TARGET_DIR, exist_ok=True) 
    
    if os.path.exists(TEMP_WORK_DIR):
        print(f"🗑️ 发现旧的临时工作目录 {TEMP_WORK_DIR}，正在删除...")
        try:
            shutil.rmtree(TEMP_WORK_DIR)
        except Exception as e:
            print(f"❌ 警告：无法删除旧的临时目录。请手动删除。错误: {e}", file=sys.stderr)
            sys.exit(1)
            
    os.makedirs(TEMP_WORK_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    print(f"✅ 临时工作目录: {TEMP_WORK_DIR}")
    print(f"✅ 最终报告将保存到目录: {FINAL_REPORT_TARGET_DIR}")


    data_files = {
        "train": "casehold_train.csv",
        "validation": "casehold_validation.csv",
        "test": "casehold_test.csv"
    }

    print(f"模型基准: allenai/longformer-base-4096")

    num_labels = 5
    MODEL_CHECKPOINT = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    try:
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载 CSV 数据集。请检查文件路径是否正确: {data_files.values()}。错误: {e}", file=sys.stderr)
        sys.exit(1)

    limited_datasets = DatasetDict({
        split_name: dataset.select(range(min(LIMIT, len(dataset))))
        for split_name, dataset in raw_datasets.items()
    })

    print("--- 正在使用 CaseHOLD MCQA 格式对数据集进行分词和重塑 ---")
    tokenized_datasets = limited_datasets.map(
        preprocess_function_mcqa,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 6. 恢复设置 Torch 格式
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels']) 
    print("✅ 数据集格式设置为 Torch，序列已预填充至等长。")


    # 7. 模型加载 
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)
    
    # 8. 定义训练参数 (包含 OOM 优化)
    training_args = TrainingArguments(
        output_dir=TEMP_WORK_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=1,  # OOM 优化：最小化实际批量大小
        per_device_eval_batch_size=1,   # OOM 优化：最小化评估批量大小
        gradient_accumulation_steps=8,  # OOM 优化：模拟 8 的逻辑批量大小
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",  
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=LOGGING_DIR,
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        fp16=True, # 🚀 修复点 4: 强制启用半精度训练
    )

    # 9. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mcqa,
    )

    # 10. 开始训练
    print("\n--- 🚀 开始模型训练 (CaseHOLD MCQA 任务) ---")
    trainer.train()
    print("--- 训练完成 ---")

    # 11. 在测试集上评估并提取指标
    print("\n--- 📊 在测试集上评估 (CaseHOLD MCQA 任务) ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    accuracy = results.get('eval_accuracy', 0.0)
    precision = results.get('eval_precision', 0.0)
    recall = results.get('eval_recall', 0.0)
    f1 = results.get('eval_f1', 0.0)

    print("最终测试集评估结果:")
    print(results)

    # 12. 保存最终结果到 TXT 文件 
    final_text_content = f"""
--- CaseHOLD MCQA 评估报告 ---
模型基准: {MODEL_CHECKPOINT}
数据集: CaseHOLD (MCQA)
测试集样本数: {len(tokenized_datasets["test"])}
---------------------------
准确率 (Accuracy): {accuracy:.4f}
精确度 (Precision - Weighted): {precision:.4f}
召回率 (Recall - Weighted): {recall:.4f}
F1 Score (Weighted): {f1:.4f}
---------------------------
"""
    try:
        with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
            f.write(final_text_content.strip())
        print(f"\n✅ 最终指标报告已保存到: {FINAL_REPORT_TXT}")
    except Exception as e:
        print(f"\n❌ 警告：无法写入最终结果文件 {FINAL_REPORT_TXT}。错误: {e}", file=sys.stderr)

    # 13. 保存详细结果到 JSON 文件
    evaluation_json_report = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "test_set_size": len(tokenized_datasets["test"]),
        "evaluation_metrics": results,
        "best_model_path_temp": trainer.state.best_model_checkpoint
    }

    try:
        with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
            json.dump(evaluation_json_report, f, indent=4)
        print(f"✅ 详细评估结果 JSON 已保存到: {FINAL_REPORT_JSON}")
    except Exception as e:
        print(f"\n❌ 警告：无法写入评估结果 JSON 文件。错误: {e}", file=sys.stderr)
    
    
    # --- 14. 关键清理步骤：删除所有中间结果 ---
    if os.path.exists(TEMP_WORK_DIR):
        print(f"\n🗑️ 正在删除临时工作目录及其所有内容: {TEMP_WORK_DIR}")
        try:
            shutil.rmtree(TEMP_WORK_DIR)
            print("✅ 所有中间文件和模型检查点已彻底清除。")
        except Exception as e:
            print(f"❌ 严重警告：无法删除临时工作目录。请手动删除 {TEMP_WORK_DIR}。错误: {e}", file=sys.stderr)
    
    print("\n✅ 程序执行完毕，仅保留文件:")
    print(f"- {FINAL_REPORT_TXT}")
    print(f"- {FINAL_REPORT_JSON}")