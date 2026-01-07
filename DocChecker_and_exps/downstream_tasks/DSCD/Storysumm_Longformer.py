# DSCD_StorySumm_Longformer.py
"""
使用的数据集：StorySumm
使用的基准："allenai/longformer-base-4096"
"""
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch
import json
import os


# --- 1. 数据加载和映射函数 ---

def load_and_map_storysumm(file_path: str) -> Dataset:
    """
    加载 StorySumm JSON 文件并将其映射为 DSCD 任务的格式。

    StorySumm 标签映射:
    0 (Unfaithful) -> 1 (Contradictory/有矛盾)
    1 (Faithful)   -> 0 (Non-Contradictory/无矛盾)
    """
    try:
        # 尝试使用 load_dataset('json', ...) 加载 (保留以防您文件格式更改)
        raw_datasets = load_dataset('json', data_files={'train': file_path}, split='train')
    except Exception as e:
        print(f"Initial load failed: {e}. Trying manual dictionary loading...")

        # --- 针对 StorySumm 顶级字典结构的修正 ---
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 假设整个文件是一个顶级字典
                data_dict = json.load(f)
        except Exception as load_err:
            print(f"FATAL ERROR: Failed to load file as a dictionary: {load_err}")
            raise load_err

        # 将字典的 values 提取为列表，并添加 ID 字段
        flattened_list = []
        for story_id, example_data in data_dict.items():
            example_data['story-id'] = story_id  # 确保 story-id 字段存在
            flattened_list.append(example_data)

        # 现在 flattened_list 是一个字典列表，可以使用 Dataset.from_list()
        raw_datasets = Dataset.from_list(flattened_list)

    # 转换和映射数据集
    def map_storysumm_to_dscd(example):
        original_label = example['label']
        # 注意: 'summary' 在 StorySumm 中通常是一个句子列表
        summary_text = " ".join(example['summary']) if isinstance(example['summary'], list) else example['summary']

        return {
            'document': summary_text,
            'label': 1 if original_label == 0 else 0,  # 标签反转: 不忠实(0) -> 有矛盾(1)
            'original_story_id': example.get('story-id', 'N/A')
        }

    # 注意：我们保留 'label' 列名，而不是 'labels'，这与您的 GPT 推理代码一致
    mapped_dataset = raw_datasets.map(map_storysumm_to_dscd, remove_columns=raw_datasets.column_names)

    # 手动创建 features 以确保 'label' 是 ClassLabel 类型，有助于后续评估
    # 但在 GPT 推理场景下，保留为 int 即可

    return mapped_dataset


# --- 2. 预处理函数 (分词) ---

def preprocess_function(examples, tokenizer):
    """
    对 StorySumm 的 'document' 文本进行分词。
    """
    return tokenizer(
        examples["document"],
        truncation=True,
        max_length=4096
    )


def compute_metrics(eval_pred):
    """计算评估指标（准确率）"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


# --- 3. 核心执行块 ---

if __name__ == '__main__':
    # 限制的记录数（用于快速测试）
    LIMIT = 1000

    # 您的文件路径
    STORYSUMM_FILE_PATH = "../../../datasets/data_from_related_work/RRC-DSCD/data/raw_data/storysumm.json"

    # 1. 定义类别数
    num_labels = 2

    # 2. 选择和加载模型/分词器
    # 仍推荐 Longformer 处理长文本 (LLM 生成的摘要可能很长)
    MODEL_CHECKPOINT = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 3. 加载、映射和划分数据集
    print("🌍 正在加载 StorySumm 数据集并映射为 DSCD 任务...")
    raw_dataset = load_and_map_storysumm(STORYSUMM_FILE_PATH)

    print(f"⚠️ 原始数据有 {len(raw_dataset)} 条，正在进行训练/验证/测试集划分 (80/10/10)...")

    # 划分数据集 (80/10/10)
    train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_test_split['train']
    test_dataset = train_test_split['test']

    train_val_split = train_val.train_test_split(test_size=0.125, seed=42)  # 10% / 80% = 0.125
    train_dataset = train_val_split['train']
    validation_dataset = train_val_split['test']

    # 将划分结果合并到 DatasetDict 并应用 LIMIT
    limited_datasets = DatasetDict({
        "train": train_dataset.select(range(min(LIMIT, len(train_dataset)))),
        "validation": validation_dataset.select(range(min(LIMIT // 8, len(validation_dataset)))),
        "test": test_dataset.select(range(min(LIMIT // 8, len(test_dataset)))),
    })

    # 4. 文本预处理 (分词)
    print("⏳ 正在进行文本分词...")
    tokenized_datasets = limited_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 5. 设置数据格式
    print("🔄 正在将标签列从 'label' 重命名为 'labels'...")
    # ------------------------------------------------------------------
    # 关键修复点：添加重命名步骤
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # ------------------------------------------------------------------

    # 此时 tokenized_datasets 包含: ['labels', 'document', 'original_story_id', 'input_ids', 'attention_mask']

    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    print(f"✅ 数据集加载、分词、划分完成 (类别数: {num_labels})。")
    print(tokenized_datasets)

    # 6. 模型加载 (使用 Longformer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)

    # 7. 定义训练参数
    training_args = TrainingArguments(
        output_dir="./storysumm_dscd_results",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./storysumm_dscd_logs',
        logging_steps=50,
        load_best_model_at_end=True,
        # fp16=True, # 如果GPU支持，推荐开启
    )

    # 8. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 9. 开始训练
    print("\n--- 🚀 开始 DSCD 模型训练 ---")
    trainer.train()

    # 10. 在测试集上评估
    print("\n--- 📊 在测试集上评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(results)