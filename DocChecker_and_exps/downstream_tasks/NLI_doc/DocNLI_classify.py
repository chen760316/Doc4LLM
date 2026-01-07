# DocNLI_classify.py
"""
使用的数据集：DOCNLI
使用的基准："allenai/longformer-base-4096"
限制选取前1000条数据
"""

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch  # 导入 torch 用于后续的张量处理


# 预处理函数 (负责文本分词)
def preprocess_function(examples, tokenizer):
    # 假设您的 JSON 文件中的字段名是 'premise' 和 'hypothesis'
    return tokenizer(
        examples["premise"],  # 文档（前提）
        examples["hypothesis"],  # 假设
        truncation=True,
        max_length=4096
    )


# 标签映射函数 (负责将字符串标签转换为整数)
def map_labels(examples, label_to_id):
    # 假设标签字段是 'label'
    examples['label'] = [label_to_id[lbl] for lbl in examples['label']]
    return examples


def compute_metrics(eval_pred):
    # 此函数用于训练过程中的评估
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # 动态加载 accuracy metric
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


# ------------------- 核心执行块 -------------------
if __name__ == '__main__':
    # 限制的记录数
    LIMIT = 1000

    # 您的文件路径（JSON 格式）
    data_files = {
        "train": "../../../datasets/data_from_related_work/DOCNLI/Data/DocNLI/train.json",
        "validation": "../../../datasets/data_from_related_work/DOCNLI/Data/DocNLI/dev.json",
        "test": "../../../datasets/data_from_related_work/DOCNLI/Data/DocNLI/test.json"
    }

    # 1. 定义标签映射和类别数 (已确认为二分类)
    label_to_id = {
        'entailment': 0,        # 正例：蕴含
        'not_entailment': 1     # 反例：不蕴含
    }
    num_labels = len(label_to_id) # 类别数 = 2

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

    # 6. 标签预处理 (字符串 -> 整数 ID)
    tokenized_datasets = tokenized_datasets.map(
        map_labels,
        batched=True,
        num_proc=4,
        fn_kwargs={"label_to_id": label_to_id}
    )

    # 检查并设置正确的格式（将 labels 列设置为 PyTorch tensor 类型）
    # 这一步是确保数据加载器可以正确处理数据的关键
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

    print(f"✅ 数据集加载、限制、分词并完成标签映射 (类别数: {num_labels})。")
    print(tokenized_datasets)

    # 7. 模型加载 (使用 num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)

    # 8. 定义训练参数
    training_args = TrainingArguments(
        output_dir="./docnli_results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./docnli_logs',
        logging_steps=500,
        load_best_model_at_end=True,
        # 推荐：如果内存不足，可以尝试使用 fp16
        # fp16=True,
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
    print("\n--- 开始模型训练 ---")
    trainer.train()

    # 11. 在测试集上评估
    print("\n--- 在测试集上评估 ---")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(results)