# DSCD_knowledge_conflict_Longformer.py
"""
使用的数据集：knowledge_conflict
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
from typing import Dict, Any, List, Optional

# --- 0. 结果保存路径定义 ---
REPORT_DIR = "report"
REPORT_FILE_NAME = "knowledge_conflict_Longformer_report.txt"
REPORT_FILE_PATH = os.path.join(REPORT_DIR, REPORT_FILE_NAME)
MODEL_OUTPUT_BASE = os.path.join(REPORT_DIR, "knowledge_conflict_Longformer_output")
LOGGING_DIR = os.path.join(MODEL_OUTPUT_BASE, "logs")
RESULTS_JSON_FILE = os.path.join(MODEL_OUTPUT_BASE, "evaluation_metrics.json")
KNOWLEDGE_CONFLICT_FILE_PATH = "sampled_data.json"  # 明确文件路径变量
# 新增：用于记录最终指标的 TXT 文件路径（在当前目录）
FINAL_RESULT_TXT = "result.txt"


# --- 1. 数据加载和映射函数 (针对 knowledge_conflict 结构重写) ---

def load_and_map_knowledge_conflict(file_path: str) -> Optional[Dataset]:
    """
    加载 knowledge_conflict JSON 文件 (列表结构) 并映射为 DSCD 任务的格式。

    映射规则：
    - 冲突样本 (Label 1): question_about_conflicting_segments + parametric_knowledge + conflicting_knowledge
    - 非冲突样本 (Label 0): question_about_nonconflicting_segments + parametric_knowledge
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        if not isinstance(data_list, list):
            raise ValueError("File content is not a list (expected a list of records).")
    except FileNotFoundError:
        print(f"FATAL ERROR: knowledge_conflict file not found at {file_path}", file=sys.stderr)
        return None
    except Exception as load_err:
        print(f"FATAL ERROR: Failed to load or parse file: {load_err}", file=sys.stderr)
        return None

    flattened_list = []

    for i, example_data in enumerate(data_list):
        unique_id = example_data.get('main_entity', f'ID_{i}')

        pk = example_data.get('parametric_knowledge', "")
        ck = example_data.get('conflicting_knowledge', "")
        qq_c = example_data.get('question_about_conflicting_segments', "")
        qq_nc = example_data.get('question_about_nonconflicting_segments', "")

        # 1. 构造冲突样本 (Label 1)
        doc_conflict = " ".join([qq_c, pk, ck])
        flattened_list.append({
            'document': doc_conflict,
            'label': 1,  # 1: 冲突 (Contradictory)
            'original_id': f"{unique_id}_C"
        })

        # 2. 构造非冲突样本 (Label 0)
        doc_non_conflict = " ".join([qq_nc, pk])
        flattened_list.append({
            'document': doc_non_conflict,
            'label': 0,  # 0: 非冲突 (Non-Contradictory)
            'original_id': f"{unique_id}_NC"
        })

    print(f"原始 {len(data_list)} 条记录扩展为 {len(flattened_list)} 条 DSCD 样本 (1:1 冲突/非冲突)")
    return Dataset.from_list(flattened_list)


# --- 2. 预处理函数 (分词) ---

def preprocess_function(examples, tokenizer):
    """
    对 knowledge_conflict 的 'document' 文本进行分词。
    """
    return tokenizer(
        examples["document"],
        truncation=True,
        max_length=4096
    )


# --- 核心修改：计算所有指标 (修复 TypeError) ---
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

    # 只传递 predictions 和 references，避免 Accuracy 的 TypeError
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
    LIMIT = 1000

    # 1. 创建输出目录
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_BASE, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    # 2. 设置输出重定向
    # 恢复原始的 stdout，以便在函数开始前打印一些关键信息到控制台
    original_stdout = sys.stdout
    print(f"🌍 正在加载 knowledge_conflict 数据集...")
    print(f"评估结果、训练日志和模型检查点将保存到: {MODEL_OUTPUT_BASE}")
    print(f"最终评估结果报告将写入文件: {REPORT_FILE_PATH}")

    try:
        with open(REPORT_FILE_PATH, 'w', encoding='utf-8') as f:
            sys.stdout = f  # 重定向 stdout 到报告文件

            # 打印起始信息到报告
            print(f"--- knowledge_conflict Longformer Fine-tuning Report ---")
            print(f"Output Directory: {MODEL_OUTPUT_BASE}")
            print("-" * 60)
            print(f"Limit applied: {LIMIT}")

            # 3. 定义类别数
            num_labels = 2

            # 4. 选择和加载模型/分词器
            MODEL_CHECKPOINT = "allenai/longformer-base-4096"
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

            # 5. 加载、映射和划分数据集
            raw_dataset = load_and_map_knowledge_conflict(KNOWLEDGE_CONFLICT_FILE_PATH)

            if raw_dataset is None:
                sys.exit(1)

            print(f"原始数据总数: {len(raw_dataset)}。正在进行训练/验证/测试集划分 (80/10/10)...")

            # 划分数据集 (80/10/10)
            train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
            train_val = train_test_split['train']
            test_dataset = train_test_split['test']

            train_val_split = train_val.train_test_split(test_size=0.125, seed=42)
            train_dataset = train_val_split['train']
            validation_dataset = train_val_split['test']


            # 将划分结果合并到 DatasetDict 并应用 LIMIT
            def limit_dataset(ds, limit, fraction=1):
                # 确保在 LIMIT 模式下，各集合大小按比例分配
                current_limit = limit if limit is None else max(1, limit // fraction)
                return ds.select(range(min(current_limit, len(ds)))) if current_limit and current_limit < len(ds) else ds


            limited_datasets = DatasetDict({
                "train": limit_dataset(train_dataset, LIMIT, fraction=1),
                "validation": limit_dataset(validation_dataset, LIMIT, fraction=8),
                "test": limit_dataset(test_dataset, LIMIT, fraction=8),
            })

            print(f"当前使用的样本数:")
            print(f"  Train size: {len(limited_datasets['train'])}, Validation size: {len(limited_datasets['validation'])}, Test size: {len(limited_datasets['test'])}")

            # 6. 文本预处理 (分词)
            print("\n⏳ 正在进行文本分词...")
            tokenized_datasets = limited_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=4,
                fn_kwargs={"tokenizer": tokenizer}
            )

            # 7. 设置数据格式
            print("🔄 正在将标签列从 'label' 重命名为 'labels'...")
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

            print(f"✅ 数据集加载、分词、划分完成 (类别数: {num_labels})。")

            # 8. 模型加载 (使用 Longformer)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)

            # 9. 定义训练参数
            training_args = TrainingArguments(
                output_dir=MODEL_OUTPUT_BASE,
                learning_rate=2e-5,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=4,
                num_train_epochs=3,
                weight_decay=0.01,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_dir=LOGGING_DIR,
                logging_steps=50,
                load_best_model_at_end=True,
                report_to="none"
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
            print("--- 训练完成 ---")

            # 12. 在测试集上评估并保存结果
            print("\n--- 📊 在测试集上评估并保存结果 ---")
            results = trainer.evaluate(tokenized_datasets["test"])
            print("最终测试集评估结果:")
            print(results)

            # 提取关键指标 (键名由 compute_metrics 函数的返回键名和 eval_ 前缀组成)
            accuracy = results.get('eval_accuracy', 0.0)
            precision = results.get('eval_precision', 0.0)
            recall = results.get('eval_recall', 0.0)
            f1 = results.get('eval_f1', 0.0)

            # 保存详细报告到 JSON 文件
            evaluation_report = {
                "model_checkpoint": MODEL_CHECKPOINT,
                "dataset_size_raw": len(raw_dataset),
                "limit_applied": LIMIT,
                "test_set_size_evaluated": len(tokenized_datasets["test"]),
                "evaluation_metrics": results,
                "best_model_checkpoint": trainer.state.best_model_checkpoint
            }

            try:
                with open(RESULTS_JSON_FILE, 'w', encoding='utf-8') as json_f:
                    json.dump(evaluation_report, json_f, indent=4)
                print(f"\n✅ 详细评估结果 JSON 已保存到: {RESULTS_JSON_FILE}")
            except Exception as e:
                print(f"\n❌ 警告：无法写入评估结果 JSON 文件。错误: {e}", file=sys.stderr)

            # 13. 保存最终模型 (可选)
            final_model_save_path = os.path.join(MODEL_OUTPUT_BASE, "final_model")
            trainer.save_model(final_model_save_path)
            print(f"✅ 最终模型已保存到: {final_model_save_path}")

    except Exception as e:
        # 在发生错误时恢复 stdout
        sys.stdout = original_stdout
        print(f"FATAL ERROR: An unexpected error occurred during execution: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        # 14. 恢复 stdout 并写入 result.txt
        sys.stdout = original_stdout

        # --- 新增功能：保存到 result.txt 文件 ---
        result_text_content = f"""
--- 模型评估结果 ---
模型检查点: {MODEL_CHECKPOINT}
数据集: knowledge_conflict
测试集样本数: {len(tokenized_datasets.get('test', []))}
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
            print(f"\n✅ 最终指标报告已保存到: {FINAL_RESULT_TXT} (当前目录)")
        except Exception as e:
            print(f"\n❌ 警告：无法写入最终结果文件 {FINAL_RESULT_TXT}。错误: {e}", file=sys.stderr)

        print(f"\n✅ DSCD knowledge_conflict Longformer 评估和训练完成。详细报告已保存到 {REPORT_FILE_PATH}")