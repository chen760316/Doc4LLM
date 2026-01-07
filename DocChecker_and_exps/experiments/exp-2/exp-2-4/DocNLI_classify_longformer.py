# DocNLI_classify.py
"""
使用的数据集：DOCNLI (二分类 NLI)
使用的基准："allenai/longformer-base-4096"
功能：计算并保存 Accuracy, Precision, Recall, F1 四个指标 (二分类 binary)。
"""
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import numpy as np
import evaluate
import torch
import os
import sys
import json
from typing import Dict, List, Any

# --- 0. 结果保存路径定义 ---
OUTPUT_DIR = "./docnli_results"
LOGGING_DIR = './docnli_logs'
# 最终报告文件路径
FINAL_REPORT_TXT = os.path.join(OUTPUT_DIR, "report/DocNLI_classify_longformer_report.txt")
FINAL_REPORT_JSON = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")


# 预处理函数 (负责文本分词)
def preprocess_function(examples, tokenizer):
    """对前提和假设进行分词和编码。"""
    return tokenizer(
        examples["premise"],  # 文档（前提）
        examples["hypothesis"],  # 假设
        truncation=True,
        max_length=4096
    )


# 标签映射函数 (负责将字符串标签转换为整数)
def map_labels(examples: Dict[str, List[Any]], label_to_id: Dict[str, int]) -> Dict[str, List[int]]:
    """将字符串标签映射为整数 ID，并将列名改为 'labels'。"""

    try:
        # 将原始的 'label' 字符串列表映射为整数 ID 列表
        # 使用一个新的键名 'labels' 避免混淆，Trainer 期望 'labels'
        examples['labels'] = [label_to_id[lbl] for lbl in examples['label']]
    except KeyError as e:
        raise KeyError(f"数据集中存在未知的标签: {e}. 请检查您的JSON文件中的标签是否与 LABEL_MAPPING 匹配。")

    # 返回一个新的字典，包含新的 'labels' 列
    return {'labels': examples['labels']}


# 🎯 修复后的指标计算函数：计算四项指标 (二分类 binary)
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """计算准确率、精确度、召回率和 F1 分数 (使用 binary average)"""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)

    # 组合多个指标
    metric = evaluate.combine([
        evaluate.load("accuracy"),
        # DOCNLI 是二分类，使用 'binary' average
        evaluate.load("precision", average="binary"),
        evaluate.load("recall", average="binary"),
        evaluate.load("f1", average="binary")
    ])

    # 只传递 predictions 和 references
    results = metric.compute(
        predictions=predictions,
        references=labels,
    )

    # 返回统一的键名
    return {
        "accuracy": results.get("accuracy", 0.0),
        "precision": results.get("precision", 0.0),
        "recall": results.get("recall", 0.0),
        "f1": results.get("f1", 0.0)
    }


# ------------------- 核心执行块 -------------------
if __name__ == '__main__':
    # 限制的记录数
    LIMIT = 1000
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    # 您的文件路径（JSON 格式）
    data_files = {
        "train": "../../../datasets/data_from_related_work/DOCNLI/Data/DocNLI/train.json",
        "validation": "../../../datasets/data_from_related_work/DOCNLI/Data/DocNLI/dev.json",
        "test": "../../../datasets/data_from_related_work/DOCNLI/Data/DocNLI/test.json"
    }

    # 1. 定义标签映射和类别数 (二分类)
    # ⚠️ 保持标签映射与代码中的一致性，注意 train/validation/test 数据集中的实际标签
    label_to_id = {
        'entailment': 0,  # 正例：蕴含
        'not_entailment': 1  # 反例：不蕴含
    }
    num_labels = len(label_to_id)  # 类别数 = 2

    print(f"✅ 结果将保存到目录: {OUTPUT_DIR}")
    MODEL_CHECKPOINT = "allenai/longformer-base-4096"
    print(f"模型基准: {MODEL_CHECKPOINT} (Max length 4096)")

    # 2. 选择和加载模型/分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 3. 加载数据集（JSON 格式）
    try:
        raw_datasets = load_dataset("json", data_files=data_files)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载 JSON 数据集。请检查文件路径是否正确。错误: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. 选取每个 split 的前 LIMIT 条数据
    limited_datasets = DatasetDict({
        split_name: dataset.select(range(min(LIMIT, len(dataset))))
        for split_name, dataset in raw_datasets.items()
    })

    # 5. 文本预处理 (分词)
    print("--- 正在进行文本分词 (Longformer 4096) ---")
    tokenized_datasets = limited_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 6. 标签预处理 (字符串 -> 整数 ID)
    print("--- 正在进行标签映射 (entailment/not_entailment -> 0/1) ---")
    tokenized_datasets = tokenized_datasets.map(
        map_labels,
        batched=True,
        num_proc=4,
        fn_kwargs={"label_to_id": label_to_id}
    )

    # 7. 设置正确的格式
    # 移除原始的 'premise', 'hypothesis' 和旧的 'label' 列
    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "label"])
    # 现在数据集应该只有 input_ids, attention_mask, 和新创建的 labels 列
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    print(f"✅ 数据集加载、分词、映射并完成格式设置 (类别数: {num_labels})。")
    print(tokenized_datasets)

    # 8. 模型加载 (使用 num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)

    # 9. 定义训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=LOGGING_DIR,
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    # 10. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # 使用计算四项指标的函数
    )

    # 11. 开始训练
    print("\n--- 🚀 开始模型训练 (DOCNLI 二分类 - Longformer) ---")
    trainer.train()
    print("--- 训练完成 ---")

    # 12. 在测试集上评估并提取指标
    print("\n--- 📊 在测试集上评估 (DOCNLI 二分类) ---")
    results = trainer.evaluate(tokenized_datasets["test"])

    # 提取评估指标
    # 注意：这里的键名是 'eval_' + compute_metrics 返回的键名
    accuracy = results.get('eval_accuracy', 0.0)
    precision = results.get('eval_precision', 0.0)
    recall = results.get('eval_recall', 0.0)
    f1 = results.get('eval_f1', 0.0)

    print("最终测试集评估结果:")
    print(results)

    # 13. 保存最终结果到 TXT 文件
    final_text_content = f"""
--- DOCNLI 二分类评估报告 ---
模型基准: {MODEL_CHECKPOINT}
数据集: DOCNLI
分类任务: 二分类 NLI (entailment:0, not_entailment:1)
测试集样本数: {len(tokenized_datasets["test"])}
---------------------------
准确率 (Accuracy): {accuracy:.4f}
精确度 (Precision - Binary): {precision:.4f}
召回率 (Recall - Binary): {recall:.4f}
F1 Score (Binary): {f1:.4f}
---------------------------
"""
    try:
        # 保存到 output 目录下的 TXT 文件
        with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
            f.write(final_text_content.strip())
        print(f"\n✅ 最终指标报告已保存到: {FINAL_REPORT_TXT}")
    except Exception as e:
        print(f"\n❌ 警告：无法写入最终结果文件 {FINAL_REPORT_TXT}。错误: {e}", file=sys.stderr)

    # 14. 保存详细结果到 JSON 文件
    evaluation_json_report = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "test_set_size": len(tokenized_datasets["test"]),
        "evaluation_metrics": results,
        "label_mapping": label_to_id,
        "best_model_path": trainer.state.best_model_checkpoint
    }

    try:
        with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
            json.dump(evaluation_json_report, f, indent=4)
        print(f"✅ 详细评估结果 JSON 已保存到: {FINAL_REPORT_JSON}")
    except Exception as e:
        print(f"\n❌ 警告：无法写入评估结果 JSON 文件。错误: {e}", file=sys.stderr)

    # 15. 保存最终模型 (可选)
    final_model_save_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_save_path)
    print(f"✅ 最终模型已保存到: {final_model_save_path}")