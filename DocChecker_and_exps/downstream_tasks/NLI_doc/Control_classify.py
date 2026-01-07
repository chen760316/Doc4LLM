# DocNLI_classify_ternary.py
"""
使用的数据集：ConTRoL
使用的基准："allenai/longformer-base-4096"
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

# 预处理函数 (负责文本分词)
def preprocess_function(examples, tokenizer):
    """对前提和假设进行分词和编码。"""
    return tokenizer(
        examples["premise"],  # 文档（前提）
        examples["hypothesis"],  # 假设
        truncation=True,
        max_length=4096 # Longformer 的最大长度
    )


# 标签映射函数 (负责将字符串标签转换为整数)
def map_labels(examples, label_to_id):
    """将字符串标签映射为整数 ID。"""
    # 确保标签在映射中存在，否则抛出错误
    try:
        examples['label'] = [label_to_id[lbl] for lbl in examples['label']]
    except KeyError as e:
        # 捕获标签错误，方便调试
        raise KeyError(f"数据集中存在未知的标签: {e}. 请检查您的JSON文件中的标签是否与 LABEL_MAPPING 匹配。")
    return examples


def compute_metrics(eval_pred):
    """计算准确率"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 动态加载 accuracy metric
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


# ------------------- 核心执行块 -------------------
if __name__ == '__main__':
    # 限制的记录数
    LIMIT = 750

    # 您的文件路径（ConTRoL 使用 JSONL 格式）
    data_files = {
        "train": "../../../datasets/data_from_related_work/ConTRoL/train.jsonl",
        "validation": "../../../datasets/data_from_related_work/ConTRoL/dev.jsonl",
        "test": "../../../datasets/data_from_related_work/ConTRoL/test.jsonl"
    }

    # 1. 定义标签映射和类别数 (改为三分类)
    # 键是数据中的字符串标签，值是 PyTorch/Trainer 期望的整数 ID
    label_to_id = {
        'c': 0,  # 矛盾
        'n': 1,  # 中性
        'e': 2  # 蕴含
    }
    num_labels = len(label_to_id)  # 类别数 = 3

    # 2. 选择和加载模型/分词器
    MODEL_CHECKPOINT = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 3. 加载数据集（JSON 格式）
    raw_datasets = load_dataset("json", data_files=data_files)

    # 4. 选取每个 split 的前 LIMIT 条数据
    limited_datasets = DatasetDict({
        split_name: dataset.select(range(LIMIT))
        for split_name, dataset in raw_datasets.items()
    })

    # 5. 文本预处理 (分词)
    tokenized_datasets = limited_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 6. 标签预处理 (字符串 -> 整数 ID) - ⚠️ 恢复此步骤！
    tokenized_datasets = tokenized_datasets.map(
        map_labels,
        batched=True,
        num_proc=4,
        fn_kwargs={"label_to_id": label_to_id}
    )

    # 检查并设置正确的格式
    # 确保 'label' 列是 PyTorch long tensor (整数类型)
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

    # 将标签列名从 'label' 改为 'labels' (Trainer默认期望)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # ⚠️ 移除原先的 convert_label_to_long 函数及其调用，因为 map_labels + set_format 已经完成任务。

    print(f"✅ 数据集加载 (JSONL)、限制、分词并完成标签映射 (类别数: {num_labels})。")
    print(tokenized_datasets)

    # 7. 模型加载 (使用 num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)

    # 8. 定义训练参数
    training_args = TrainingArguments(
        output_dir="./Control_results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./Control_logs',
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # fp16=True, # 可选
    )

    # 9. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 10. 开始训练
    print("\n--- 开始模型训练 (三元分类) ---")
    trainer.train()

    # 11. 在测试集上评估
    print("\n--- 在测试集上评估 (三元分类) ---")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(results)