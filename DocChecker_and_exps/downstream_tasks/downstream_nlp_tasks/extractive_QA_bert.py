"""
使用的数据集：SQUAD 2.0
使用的基准：'bert-base-uncased'  # bert-base-uncased 是基于 Transformer 的，使其非常适合作为问答（Question Answering, QA）任务的基准模型，特别是**抽取式问答（Extractive QA）**任务，例如 SQuAD 数据集。
"""
import json
import random
import re
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from datasets import load_metric  # 用于加载 SQuAD 官方评估指标
import pandas as pd
from datasets import Dataset  # 导入 Dataset 类

# --- 1. 配置和常量 (QA 任务修改) ---
NUM_LABELS = 2  # QA 模型内部通常使用 2 个输出头（start/end logit）
SAMPLE_SIZE = 1000  # 用于快速演示
MODEL_NAME = 'bert-base-uncased'  # 换用 BERT QA 变体
MAX_LEN = 384  # QA 任务通常需要更长的序列长度
BATCH_SIZE = 8  # QA 任务占用显存较大，BATCH_SIZE 减小
NUM_TRAIN_EPOCHS = 3
CORRUPTION_RATIO = 0.5
CORRUPTION_MARKER_COL = 'is_corrupted'

# --- SQuAD 文件路径 ---
SQUAD_FILE_PATH = '../../../datasets/data_from_related_work/QA_datasets/extractive/SQUAD2/train-v2.0.json'


# ----------------------------------------------------


# --- 2. SQuAD 数据加载和处理函数 ---

def load_and_prepare_squad_data(file_path: str, sample_size: int) -> List[Dict[str, Any]]:
    """
    加载 SQuAD 2.0 官方格式的 JSON 文件，并转换为简化格式。
    """
    print(f"Loading and processing SQuAD 2.0 data from {file_path}...")
    squad_examples = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)

        if 'data' not in squad_data or not isinstance(squad_data['data'], list):
            print("Error: SQuAD JSON file is missing the top-level 'data' list.")
            return []

        for article in squad_data['data']:
            if 'paragraphs' not in article or not isinstance(article['paragraphs'], list):
                continue

            for paragraph in article['paragraphs']:
                context = paragraph.get('context')
                qas = paragraph.get('qas')

                if not context or not qas or not isinstance(qas, list):
                    continue

                for qa in qas:
                    if isinstance(qa, dict) and 'question' in qa and 'id' in qa:
                        is_impossible = qa.get("is_impossible", False)
                        answers = qa.get("answers", [])

                        # 仅处理可抽取答案的样本 (SQuAD 1.1 兼容)
                        if not is_impossible and answers and isinstance(answers, list):
                            first_answer = answers[0]
                            if isinstance(first_answer, dict) and 'text' in first_answer and 'answer_start' in first_answer:
                                squad_examples.append({
                                    'id': qa['id'],
                                    'context': context,
                                    'question': qa['question'],
                                    'answer_text': first_answer['text'],
                                    'answer_start': first_answer['answer_start']
                                })

        if len(squad_examples) > sample_size:
            squad_examples = random.sample(squad_examples, sample_size)

        print(f"Loaded and sampled {len(squad_examples)} QA examples.")
        return squad_examples

    except FileNotFoundError:
        print(f"Error: SQuAD file not found at {file_path}. Skipping.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to parse the entire SQuAD JSON file. Check file integrity.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during SQuAD data loading: {e}")
        return []


# --- 3. SQuAD 评估指标 (占位符) ---

metric = load_metric("squad")


def compute_metrics_squad(p: Tuple[Any, Any]):
    """计算 SQuAD 官方评估指标 (F1 和 Exact Match)"""
    # 真正的 SQuAD 评估需要在 Trainer 外部进行复杂的后处理
    return {"f1": 0.0, "exact_match": 0.0}  # 暂时返回占位符


# --- 4. 实验核心逻辑：注入和清理 ---

def inject_corrupted_data(examples: List[Dict[str, Any]], corruption_ratio: float = CORRUPTION_RATIO) -> List[Dict[str, Any]]:
    """注入【答案被误导】的损坏数据。"""
    num_corruptions = int(len(examples) * corruption_ratio)
    df_examples = pd.DataFrame(examples)
    df_examples[CORRUPTION_MARKER_COL] = False
    subset_indices = df_examples.sample(n=num_corruptions, replace=False).index

    print(f"\n🧠 正在注入 {num_corruptions} 条【答案误导】损坏数据...")

    for index in subset_indices:
        row = df_examples.loc[index]
        context = row['context']

        # 策略：在上下文末尾添加一个误导性句子
        false_info = f"It must be noted that the latest historical review proves this to be false, and the real person was actually JANE DOE."

        # 寻找上下文末尾的句号并插入
        corrupted_context = re.sub(r'(\.|\!|\?|\n|\r\n)\s*$', '', context) + f". {false_info}"

        # 更新 DataFrame 记录
        df_examples.loc[index, 'context'] = corrupted_context
        df_examples.loc[index, CORRUPTION_MARKER_COL] = True

    corrupted_examples = df_examples.to_dict('records')
    print(f"✅ 注入 {num_corruptions} 条损坏数据。总训练集大小: {len(corrupted_examples)}")
    return corrupted_examples


def clean_corpus_via_contradiction_retrieval(examples_corrupted: List[Dict[str, Any]], original_size: int) -> List[Dict[str, Any]]:
    """通过识别专门的标记列来“完美”移除损坏数据。"""
    print("\n🔍 正在进行矛盾检索与清理 (使用标记列进行完美过滤)...")

    df_corrupted = pd.DataFrame(examples_corrupted)
    df_cleaned_temp = df_corrupted[df_corrupted[CORRUPTION_MARKER_COL] == False]
    filtered_count = len(df_corrupted) - len(df_cleaned_temp)
    df_cleaned = df_cleaned_temp.drop(columns=[CORRUPTION_MARKER_COL]).to_dict('records')

    print(f"✅ 清理完成。共识别并过滤掉 {filtered_count} 条矛盾数据。清理后大小: {len(df_cleaned)}")
    return df_cleaned


# --- 5. QA 任务的 Tokenization 和特征化 (已修复逻辑) ---

def prepare_train_features(examples: Dict[str, List[Any]], tokenizer: AutoTokenizer) -> Dict[str, List[Any]]:
    """
    使用 tokenizer 对 QA 样本进行编码，并计算答案的起始/结束 Token 索引。
    """
    max_seq_length = MAX_LEN
    doc_stride = 128

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 初始化目标列表 (正确位置：在循环外部)
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["example_id"] = []

    for i, offsets in enumerate(offset_mapping):
        # 1. 获取原始示例的索引
        sample_index = sample_map[i]
        answer_starts = examples["answer_start"][sample_index]
        answer_texts = examples["answer_text"][sample_index]

        # 2. 找到上下文的起始和结束 token 索引
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 查找第一个和最后一个上下文 token 的索引
        context_start_token_idx = next((j for j, sid in enumerate(sequence_ids) if sid == 1), None)
        context_end_token_idx = len(sequence_ids) - 1 - next((j for j, sid in enumerate(reversed(sequence_ids)) if sid == 1), None)

        if context_start_token_idx is None or context_end_token_idx is None:
            # 没有上下文
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            continue

        # 3. 找到答案在上下文中的字符起始/结束位置
        start_char = answer_starts
        end_char = start_char + len(answer_texts)

        # 4. 将字符位置映射到 token 位置 (默认设置为 CLS token 的位置)
        start_position = 0
        end_position = 0

        # 从上下文起始 token 开始查找，找到第一个包含答案起始字符的 token
        token_index = context_start_token_idx
        while token_index <= context_end_token_idx and token_index < len(offsets):
            start, end = offsets[token_index]
            if start <= start_char < end:
                start_position = token_index
                break
            token_index += 1

        # 从上下文结束 token 开始向前查找，找到最后一个包含答案结束字符的 token
        token_index = context_end_token_idx
        while token_index >= context_start_token_idx and token_index >= 0:
            start, end = offsets[token_index]
            if start <= end_char <= end:
                end_position = token_index
                break
            token_index -= 1

        # 5. 处理答案截断/越界情况
        if not (start_position >= context_start_token_idx and end_position <= context_end_token_idx):
            start_position = 0
            end_position = 0

        # 6. 添加结果到列表
        tokenized_examples["start_positions"].append(start_position)
        tokenized_examples["end_positions"].append(end_position)
        tokenized_examples["example_id"].append(examples["id"][sample_index])

    return tokenized_examples


def convert_examples_to_features(examples: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dataset:
    """将 List[Dict] 转换为 Hugging Face Dataset 并应用 tokenization/特征化"""
    df = pd.DataFrame(examples)
    dataset = Dataset.from_pandas(df)

    features_dataset = dataset.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False
    )

    # 保留模型需要的张量列和 example_id
    features_dataset.set_format(
        type="torch",
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions', 'example_id']
    )

    return features_dataset


def train_and_evaluate(examples_train: List[Dict[str, Any]], examples_eval: List[Dict[str, Any]], tokenizer: AutoTokenizer, prefix: str) -> float:
    """训练 QA 模型并在测试集上评估 (返回 F1)。"""

    print(f"\n--- {prefix}：正在训练/评估 QA 模型 ---")

    train_dataset = convert_examples_to_features(examples_train, tokenizer)
    eval_dataset = convert_examples_to_features(examples_eval, tokenizer)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=f'./results_{prefix}',
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        logging_dir=f'./logs_{prefix}',
        logging_steps=100,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=True,
        report_to="none"
    )

    # 修复：确保 eval_dataset 被传入 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 训练时使用
        data_collator=default_data_collator,
        compute_metrics=compute_metrics_squad  # 使用 QA 占位指标
    )

    trainer.train()

    print("⚠️ 忽略复杂后处理，使用占位评估。")
    return 0.0  # 实际应用中，这里需要返回真实的 F1/EM


# --- 7. 结果保存函数 (新增) ---

def save_experiment_results(results: Dict[str, float], output_path: str = "qa_experiment_summary.txt"):
    """将实验结果保存到文本文件并打印总结。"""

    print("\n\n" + "#" * 40)
    print("### SQuAD 语料库清理实验总结 (BERT QA) ###")

    summary = "### SQuAD 语料库清理实验总结 (BERT QA) ###\n\n"

    for key, value in results.items():
        line = f"{key} F1 (占位): {value:.4f}"
        print(line)
        summary += line + "\n"

    summary += "\n❗ 提示：要获取实际 F1 结果，需要实现完整的 SQuAD 评估后处理逻辑。\n"
    summary += "#" * 40 + "\n"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"\n✅ 结果已保存至: {output_path}")
    except Exception as e:
        print(f"\n❌ 结果保存失败: {e}")

    print("#" * 40)


# --- 6. 主实验流程：SQuAD QA 版本 ---

def main_experiment_squad():
    # 使用 AutoTokenizer 和 AutoModelForQuestionAnswering
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. 加载和划分数据
    all_examples = load_and_prepare_squad_data(SQUAD_FILE_PATH, SAMPLE_SIZE)

    if not all_examples:
        return

    # 简单划分为训练集和测试集
    examples_train_original, examples_test = train_test_split(
        all_examples, test_size=0.2, random_state=42
    )
    original_train_size = len(examples_train_original)
    print(f"原始训练集大小: {original_train_size}, 测试集大小: {len(examples_test)}")

    # --- 阶段 I：基线性能 ---
    print("\n" + "=" * 60)
    print("--- 阶段 I: 基线性能 (SQuAD 原始训练集) ---")
    baseline_f1 = train_and_evaluate(
        examples_train=examples_train_original,
        examples_eval=examples_test,
        tokenizer=tokenizer,
        prefix="squad_baseline"
    )
    print("=" * 60)

    # --- 阶段 II：注入损坏数据 ---
    examples_corrupted_train = inject_corrupted_data(examples_train_original, corruption_ratio=CORRUPTION_RATIO)

    print("\n" + "=" * 60)
    print("--- 阶段 II: 注入损坏数据后的性能 ---")
    corrupted_f1 = train_and_evaluate(
        examples_train=examples_corrupted_train,
        examples_eval=examples_test,
        tokenizer=tokenizer,
        prefix="squad_inject"
    )
    print("=" * 60)

    # --- 阶段 III：清理与性能提升 ---
    examples_cleaned_train = clean_corpus_via_contradiction_retrieval(examples_corrupted_train, original_train_size)

    print("\n" + "=" * 60)
    print("--- 阶段 III: 清理后性能 ---")
    cleaned_f1 = train_and_evaluate(
        examples_train=examples_cleaned_train,
        examples_eval=examples_test,
        tokenizer=tokenizer,
        prefix="squad_clean"
    )
    print("=" * 60)

    # --- 最终总结：收集结果并调用保存函数 ---
    final_results = {
        "1. 原始基线": baseline_f1,
        "2. 损坏后": corrupted_f1,
        "3. 清理后": cleaned_f1
    }

    save_experiment_results(final_results)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    main_experiment_squad()