"""
使用的数据集：CaseHOLD
使用的基准："allenai/longformer-base-4096"

本目录下唯一一个用于多选QA的代码评估
其余三个代码文件用于NLI设置
"""

# CaseHOLD_MCQA_classify.py (根据实际案例修改列名)

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer, EvalPrediction
import numpy as np
import evaluate
import torch

# 🎯 确保这个函数定义在 __main__ 块或使用它的地方之前
def compute_metrics_mcqa(eval_pred: EvalPrediction):
    """计算多项选择问答的准确率"""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=1)

    # 动态加载 accuracy metric
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)


def preprocess_function_mcqa(examples, tokenizer):
    """
    针对 CaseHOLD 多项选择问答任务的预处理函数。
    将 'citing_prompt' 与五个 'holding_i' 选项配对。
    """

    # 🚨 根据您提供的案例，明确列名映射
    context_name = "citing_prompt"  # 案件上下文
    num_options = 5

    # === MCQA 输入重塑 (Longformer/BERT要求) ===

    # 1. 复制 context
    contexts = [[examples[context_name][i]] * num_options for i in range(len(examples[context_name]))]

    # 2. 收集所有选项
    options = []
    for i in range(num_options):
        option_col_name = f"holding_{i}"
        try:
            option_column = examples[option_col_name]
        except KeyError:
            raise KeyError(f"缺少 CaseHOLD 选项列。请检查您的 CSV 中是否有 '{option_col_name}' 等列名。")

        options.append(option_column)

    # options: [[holding0_row1, holding0_row2, ...], [holding1_row1, holding1_row2, ...], ...]

    # 3. 解决 TypeError: can only concatenate list (not "tuple") to list
    # a) 使用 zip(*) 进行转置，得到的是元组列表
    options_transposed_of_tuples = zip(*options)

    # b) ⚠️ 将元组列表转换为列表列表
    # 这一步是关键修复
    options_transposed = [list(t) for t in options_transposed_of_tuples]

    # c) 将上下文和选项展平
    contexts = sum(contexts, [])
    options_flat = sum(options_transposed, [])  # 现在 sum() 正在连接列表和列表，不会报错

    # 4. 分词
    tokenized_examples = tokenizer(
        contexts,
        options_flat,
        truncation=True,
        max_length=4096,
        padding="max_length"
    )

    # 5. 重塑输出张量
    return {
        **{k: [v[i:i + num_options] for i in range(0, len(v), num_options)] for k, v in tokenized_examples.items()},
        "labels": examples["label"],
    }


# ------------------- 核心执行块 -------------------
if __name__ == '__main__':
    # 限制的记录数
    LIMIT = 950

    # 您的文件路径 (CaseHOLD 使用 CSV 格式)
    data_files = {
        "train": "../../../datasets/data_from_related_work/CaseHOLD/casehold_train.csv",
        "validation": "../../../datasets/data_from_related_work/CaseHOLD/casehold_validation.csv",
        "test": "../../../datasets/data_from_related_work/CaseHOLD/casehold_test.csv"
    }

    # 1. 定义标签和类别数 (MCQA 的标签是正确选项的索引 0-4)
    # CaseHOLD 是 5 选 1 任务，标签已经是整数 0, 1, 2, 3, 4
    num_labels = 5

    # 2. 选择和加载模型/分词器
    MODEL_CHECKPOINT = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 3. 加载数据集（CSV 格式）
    raw_datasets = load_dataset("csv", data_files=data_files)

    # 4. 选取每个 split 的前 LIMIT 条数据
    limited_datasets = DatasetDict({
        split_name: dataset.select(range(LIMIT))
        for split_name, dataset in raw_datasets.items()
    })

    # 5. 文本预处理 (分词)
    print("--- 正在使用 CaseHOLD MCQA 格式对数据集进行分词和重塑 ---")
    tokenized_datasets = limited_datasets.map(
        preprocess_function_mcqa,  # 使用修复后的 MCQA 预处理函数
        batched=True,
        num_proc=4,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # 6. 设置正确的格式
    # MCQA 任务只需要 input_ids, attention_mask, 和 labels (整数索引)
    tokenized_datasets.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

    print(f"✅ 数据集加载 (CSV)、限制、分词并完成重塑 (选项数: {num_labels})。")
    print(tokenized_datasets)

    # 7. 模型加载 (使用 AutoModelForMultipleChoice)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_CHECKPOINT, num_labels=num_labels)

    # 8. 定义训练参数
    training_args = TrainingArguments(
        output_dir="./CaseHOLD_MCQA_results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./CaseHOLD_MCQA_logs',
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # fp16=True,
    )

    # 9. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_mcqa,  # 使用 MCQA 评估指标
    )

    # 10. 开始训练
    print("\n--- 开始模型训练 (CaseHOLD MCQA 任务) ---")
    trainer.train()

    # 11. 在测试集上评估
    print("\n--- 在测试集上评估 (CaseHOLD MCQA 任务) ---")
    results = trainer.evaluate(tokenized_datasets["test"])
    print(results)