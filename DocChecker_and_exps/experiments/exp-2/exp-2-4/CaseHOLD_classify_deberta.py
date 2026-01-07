# CaseHOLD_MCQA_classify.py
"""
使用的数据集：CaseHOLD (多项选择问答 MCQA)
使用的基准："microsoft/deberta-v3-base"
功能：计算并保存 Accuracy, Precision, Recall, F1 四个指标 (多分类 weighted)。
"""
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer, EvalPrediction
import numpy as np
# import evaluate # ⚠️ 不再需要 evaluate 库
import torch
import os
import sys
import json
from typing import Dict
# 🎯 导入 Scikit-learn 指标计算模块 (替代 evaluate 库)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --- 0. 结果保存路径定义 ---
# 🚨 关键修复：使用 Autodl 的数据盘路径，解决磁盘空间不足问题
OUTPUT_DIR = "report/CaseHOLD_MCQA_Deberta_results"
LOGGING_DIR = 'report/CaseHOLD_MCQA_Deberta_logs' # 虽然禁用，但路径仍需定义
# 最终报告文件路径
FINAL_REPORT_TXT = os.path.join(OUTPUT_DIR, "report/CaseHOLD_classify_deberta_report.txt")
FINAL_REPORT_JSON = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")


# 🎯 最终修复后的指标计算函数：直接使用 Scikit-learn
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


def preprocess_function_mcqa(examples, tokenizer):
    """
    针对 CaseHOLD 多项选择问答任务的预处理函数。
    将 'citing_prompt' 与五个 'holding_i' 选项配对。
    """
    context_name = "citing_prompt"
    num_options = 5

    # 1. 复制 context
    contexts = [[examples[context_name][i]] * num_options for i in range(len(examples[context_name]))]

    # 2. 收集所有选项
    options = []
    for i in range(num_options):
        option_col_name = f"holding_{i}"
        try:
            option_column = examples[option_col_name]
        except KeyError:
            print(f"FATAL ERROR: 缺少 CaseHOLD 选项列。请检查您的 CSV 中是否有 '{option_col_name}' 等列名。", file=sys.stderr)
            raise

        options.append(option_column)

    # 3. 展平选项：转置并展平
    options_transposed_of_tuples = zip(*options)
    options_transposed = [list(t) for t in options_transposed_of_tuples]

    # 4. 展平上下文和选项
    contexts = sum(contexts, [])
    options_flat = sum(options_transposed, [])

    # 5. 分词
    tokenized_examples = tokenizer(
        contexts,
        options_flat,
        truncation=True,
        # ❗❗ 关键：使用 DeBERTa 默认的最大长度 512 ❗❗
        max_length=512,
        padding="max_length"
    )

    # 6. 重塑输出张量：(batch_size * num_options) -> (batch_size, num_options, sequence_length)
    return {
        **{k: [v[i:i + num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()},
        "labels": examples["label"],
    }


# ------------------- 核心执行块 -------------------
if __name__ == '__main__':
    # 限制的记录数
    LIMIT = 950
    # 确保输出目录和报告父目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "report"), exist_ok=True)


    # 您的文件路径 (CaseHOLD 使用 CSV 格式)
    data_files = {
        "train": "casehold_train.csv",
        "validation": "casehold_validation.csv",
        "test": "casehold_test.csv"
    }

    print(f"✅ 结果将保存到目录: {OUTPUT_DIR}")
    MODEL_CHECKPOINT = "microsoft/deberta-v3-base"
    print(f"模型基准: {MODEL_CHECKPOINT}")


    # 1. 定义标签和类别数
    num_labels = 5

    # 2. 选择和加载模型/分词器
    try:
        # 注意：使用 DeBERTa-v3-base 需要 trust_remote_code=True
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, trust_remote_code=True)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载 Tokenizer. 请检查网络连接或缺少 trust_remote_code=True。错误: {e}", file=sys.stderr)
        sys.exit(1)


    # 3. 加载数据集（CSV 格式）
    try:
        raw_datasets = load_dataset("csv", data_files=data_files)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载 CSV 数据集。请检查文件路径是否正确。错误: {e}", file=sys.stderr)
        sys.exit(1)


    # 4. 选取每个 split 的前 LIMIT 条数据
    limited_datasets = DatasetDict({
        split_name: dataset.select(range(min(LIMIT, len(dataset))))
        for split_name, dataset in raw_datasets.items()
    })

    # 5. 文本预处理 (分词)
    print("--- 正在使用 CaseHOLD MCQA 格式对数据集进行分词和重塑 ---")
    print(f"❗❗ 警告: {MODEL_CHECKPOINT} 最大长度为 512，长文档将被截断。")

    tokenized_datasets = limited_datasets.map(
        preprocess_function_mcqa,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 6. 设置正确的格式
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    print(f"✅ 数据集加载 (CSV)、限制、分词并完成重塑 (选项数: {num_labels})。")

    # 7. 模型加载 (使用 AutoModelForMultipleChoice)
    try:
        model = AutoModelForMultipleChoice.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels, trust_remote_code=True)
    except Exception as e:
        print(f"FATAL ERROR: 无法加载模型权重。请检查网络连接或 trust_remote_code。错误: {e}", file=sys.stderr)
        sys.exit(1)


    # 8. 定义训练参数
    # ⚠️ 关键修改：禁用所有模型和日志保存
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",  
        
        # === 禁用所有检查点和日志保存 ===
        save_strategy="no",             # 禁用所有检查点保存
        logging_dir=None,               # 禁用日志目录
        load_best_model_at_end=False,   # 禁用加载最佳模型功能（因为没有模型保存）
        # ================================
        
        logging_steps=500,
        metric_for_best_model="accuracy",
        report_to="none"
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
    print("\n--- 🚀 开始模型训练 (CaseHOLD MCQA 任务 - DeBERTa) ---")
    trainer.train()
    print("--- 训练完成 ---")


    # 11. 在测试集上评估并提取指标
    print("\n--- 📊 在测试集上评估 (CaseHOLD MCQA 任务) ---")
    # 由于 load_best_model_at_end=False，这里评估的是训练结束时的模型
    results = trainer.evaluate(tokenized_datasets["test"])

    # 提取评估指标
    accuracy = results.get('eval_accuracy', 0.0)
    precision = results.get('eval_precision', 0.0)
    recall = results.get('eval_recall', 0.0)
    f1 = results.get('eval_f1', 0.0)

    print("最终测试集评估结果:")
    print(results)


    # 12. 保存最终结果到 TXT 文件 (只保留此保存)
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
        # 保存到 output 目录下的 TXT 文件
        with open(FINAL_REPORT_TXT, 'w', encoding='utf-8') as f:
            f.write(final_text_content.strip())
        print(f"\n✅ 最终指标报告已保存到: {FINAL_REPORT_TXT}")
    except Exception as e:
        print(f"\n❌ 警告：无法写入最终结果文件 {FINAL_REPORT_TXT}。错误: {e}", file=sys.stderr)

    # 13. 保存详细结果到 JSON 文件 (可选保留)
    evaluation_json_report = {
        "model_checkpoint": MODEL_CHECKPOINT,
        "test_set_size": len(tokenized_datasets["test"]),
        "evaluation_metrics": results,
        "final_epoch": trainer.state.epoch
    }

    try:
        with open(FINAL_REPORT_JSON, 'w', encoding='utf-8') as f:
            json.dump(evaluation_json_report, f, indent=4)
        print(f"✅ 详细评估结果 JSON 已保存到: {FINAL_REPORT_JSON}")
    except Exception as e:
        print(f"\n❌ 警告：无法写入评估结果 JSON 文件。错误: {e}", file=sys.stderr)

    # 14. 禁用最终模型保存
    # final_model_save_path = os.path.join(OUTPUT_DIR, "final_model")
    # trainer.save_model(final_model_save_path) 
    # print(f"✅ 最终模型已保存到: {final_model_save_path}")