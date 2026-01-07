# DSCD_classify_nested.py
"""
使用的数据集：ContraDOC
使用的基准："allenai/longformer-base-4096"
功能：计算并保存 Accuracy, Precision, Recall, F1 四个指标。
"""
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
import json
import os
import sys
from typing import Dict, Any, List

# --- 0. 结果保存路径定义 ---
REPORT_DIR = "report/ControDOC_Longformer_report"
MODEL_OUTPUT_DIR = os.path.join(REPORT_DIR, "checkpoint")
LOGGING_DIR = os.path.join(REPORT_DIR, "logs")
RESULTS_FILE = os.path.join(REPORT_DIR, "evaluation_results.json")
# 新增：用于记录最终指标的 TXT 文件
FINAL_RESULT_TXT = "result.txt"


# --- 2. 预处理函数 ---

def preprocess_function(examples, tokenizer):
    """
    对 ContraDOC 数据集进行分词。
    """
    # Longformer 的最大长度是 4096
    return tokenizer(
        examples["document"],
        truncation=True,
        max_length=4096
    )


def compute_metrics(eval_pred):
    """
    计算评估指标（准确率、精确度、召回率、F1）。
    注意：在二分类任务中，precision, recall, f1 使用 average='binary'。
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 使用 evaluate.combine 组合多个指标，并在加载时指定配置
    metric = evaluate.combine([
        evaluate.load("accuracy"),
        # 为 precision, recall, f1 显式指定二分类 (binary)
        evaluate.load("precision", average="binary"),
        evaluate.load("recall", average="binary"),
        evaluate.load("f1", average="binary")
    ])

    # 只传递 predictions 和 references，因为 accuracy 模块不接受 'average' 参数
    results = metric.compute(
        predictions=predictions,
        references=labels,
    )

    # 确保返回的键名与 Trainer 期望的一致
    return {
        "accuracy": results.get("accuracy", 0.0),
        "precision": results.get("precision", 0.0),
        "recall": results.get("recall", 0.0),
        "f1": results.get("f1", 0.0)
    }


# --- 3. 核心执行块 ---

if __name__ == '__main__':
    # 限制的记录数（用于快速测试）
    # ⚠️ 请注意：如果 LIMIT > 0，则仅使用部分数据。若要使用所有数据，请设置 LIMIT = None 或 0
    LIMIT = 1000

    # 您的文件路径
    data_files = {
        "train": "sampled_contradoc.json"
    }

    # 1. 创建输出目录
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    print(f"✅ 结果将保存到目录: {REPORT_DIR}")

    # 2. 定义类别数
    num_labels = 2

    # 3. 选择和加载模型/分词器
    MODEL_CHECKPOINT = "allenai/longformer-base-4096"

    try:
        # 在 from_pretrained 中设置 trust_remote_code=True 以增加兼容性
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, trust_remote_code=True)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载 Tokenizer. 请检查网络连接和 HF_ENDPOINT 环境变量。\n错误详情: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. 手动加载 ContraDOC JSON 文件并展平
    print("🌍 正在手动加载 ContraDOC JSON 文件并展平结构...")
    file_path = data_files["train"]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: ContraDOC file not found at {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: 加载或解析 JSON 文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 展平数据结构
    flattened_data: List[Dict[str, Any]] = []
    if "pos" in data:
        for unique_id, doc_data in data["pos"].items():
            # 确保 'text' 键存在
            if "text" in doc_data:
                flattened_data.append({"document": doc_data["text"], "label": 1, "unique_id": unique_id})
    if "neg" in data:
        for unique_id, doc_data in data["neg"].items():
            # 确保 'text' 键存在
            if "text" in doc_data:
                flattened_data.append({"document": doc_data["text"], "label": 0, "unique_id": unique_id})

    if not flattened_data:
        print("FATAL ERROR: 数据集中没有找到有效的 'pos' 或 'neg' 样本。", file=sys.stderr)
        sys.exit(1)

    raw_dataset = Dataset.from_list(flattened_data)

    # 5. 划分数据集为 80% 训练, 10% 验证, 10% 测试
    print(f"⚠️ 原始数据有 {len(raw_dataset)} 条，正在进行训练/验证/测试集划分 (80/10/10)...")

    train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_test_split['train']
    test_dataset = train_test_split['test']

    # 10% / (80% + 10%) = 0.125
    train_val_split = train_val.train_test_split(test_size=0.125, seed=42)
    train_dataset = train_val_split['train']
    validation_dataset = train_val_split['test']


    # 限制数据集大小
    def limit_dataset(ds, limit):
        return ds.select(range(min(limit, len(ds)))) if limit and limit < len(ds) else ds


    limited_datasets = DatasetDict({
        "train": limit_dataset(train_dataset, LIMIT),
        "validation": limit_dataset(validation_dataset, LIMIT),
        "test": limit_dataset(test_dataset, LIMIT),
    })

    # 6. 文本预处理 (分词)
    print("⏳ 正在进行文本分词...")
    tokenized_datasets = limited_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,  # 使用多核加速分词
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 7. 设置数据格式
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    print(f"✅ 数据集加载、分词、划分完成 (类别数: {num_labels})。")
    print(tokenized_datasets)

    # 8. 模型加载
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels, trust_remote_code=True)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载模型权重. 请检查网络连接和依赖。\n错误详情: {e}", file=sys.stderr)
        sys.exit(1)

    # 9. 定义训练参数
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,  # 保存模型检查点
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=LOGGING_DIR,  # 保存训练日志
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none"  # 避免自动报告到外部服务
    )

    # 10. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 11. 开始训练
    print("\n--- 🚀 开始 DSCD 模型训练 ---")
    trainer.train()

    # 12. 在测试集上评估并保存结果
    print("\n--- 📊 在测试集上评估并保存结果 ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # 提取关键指标 (键名由 compute_metrics 函数的返回键名和 eval_ 前缀组成)
    accuracy = results.get('eval_accuracy', 0.0)
    precision = results.get('eval_precision', 0.0)
    recall = results.get('eval_recall', 0.0)
    f1 = results.get('eval_f1', 0.0)

    print("测试集评估结果:")
    print(results)

    # --- 新增功能：保存到 result.txt 文件 ---
    result_text_content = f"""
--- 模型评估结果 ---
模型检查点: {MODEL_CHECKPOINT}
最佳模型路径: {trainer.state.best_model_checkpoint}
数据集大小 (总): {len(raw_dataset)}
测试集样本数: {len(tokenized_datasets["test"])}
---------------------------
准确率 (Accuracy): {accuracy:.4f}
精确度 (Precision): {precision:.4f}
召回率 (Recall): {recall:.4f}
F1 Score: {f1:.4f}
---------------------------
"""
    try:
        with open(FINAL_RESULT_TXT, 'w', encoding='utf-8') as f:
            f.write(result_text_content.strip())
        print(f"\n✅ 详细评估结果已额外保存到: {FINAL_RESULT_TXT} (当前目录)")
    except Exception as e:
        print(f"\n❌ 警告：无法写入最终结果文件 {FINAL_RESULT_TXT}。错误: {e}", file=sys.stderr)

    # 保存结果到 JSON 文件
    evaluation_report = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "dataset_size": len(raw_dataset),
        "test_set_size": len(tokenized_datasets["test"]),
        "evaluation_metrics": results,
        "best_model_path": trainer.state.best_model_checkpoint
    }

    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=4)
        print(f"✅ 详细评估结果已保存到: {RESULTS_FILE}")
    except Exception as e:
        print(f"\n❌ 警告：无法写入评估结果文件 {RESULTS_FILE}。错误: {e}", file=sys.stderr)

    # 13. 保存最终模型 (可选)
    final_model_save_path = os.path.join(REPORT_DIR, "final_model")
    trainer.save_model(final_model_save_path)
    print(f"✅ 最终模型已保存到: {final_model_save_path}")