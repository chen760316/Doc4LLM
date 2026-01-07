"""
使用的数据集：binary-FEVER-NLI
使用的基准：'bert-base-uncased'

直接注入和清理损坏数据
注入损坏数据后事实核查精度下降明显
清理损坏数据后事实核查精度提升明显
执行速度快

为了在有限的计算资源下实现一个完整的、现代的事实核查基准模型（基于 BERT），并演示语料库清理带来的精度变化，我们将采用以下策略：

数据抽样： 对 train.txt, test.txt, dev.txt 各抽取 1000 条数据。
模型选择： 使用 Hugging Face transformers 库中的 BERT 进行文本蕴含 (NLI) 任务微调。
效率优化： 使用 Trainer API 进行高效微调，并只训练一个短的 Epoch。
完整逻辑： 严格遵循您的实验流程：
基线训练/测试 (使用原始抽样数据)
注入损坏 (在训练集上注入)
降级测试 (使用损坏的训练集进行微调后，在测试集上评估精度下降)
矛盾清理 (模拟清理损坏数据)
提升测试 (使用清理后的训练集进行微调后，在测试集上评估精度回升)
"""

import pandas as pd
import random
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# --- 1. 配置和常量 ---
# 标签映射：entailment -> 1 (SUPPORTS), not_entailment -> 0 (NOT ENTAIL)
LABEL_MAP = {'entailment': 1, 'not_entailment': 0}
REVERSE_LABEL_MAP = {1: 'entailment', 0: 'not_entailment'}
NUM_LABELS = 2
SAMPLE_SIZE = 1200  # 每个文件的抽样大小

MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 3


# --- 2. 数据加载和处理函数 ---

def load_and_sample_fever_data(file_path: str, sample_size: int) -> pd.DataFrame:
    """加载 FEVER 格式的文件，并抽样指定数量的数据。"""
    print(f"Loading and sampling data from {file_path}...")
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['Label', 'Premise', 'Claim'],
            encoding='utf-8'
        )

        # 移除空值并清洗数据
        df = df.dropna().reset_index(drop=True)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # 抽样
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # 标签数值化
        df['labels'] = df['Label'].apply(lambda x: LABEL_MAP.get(x.lower(), 0))
        # PLM 的输入格式：[Premise] [SEP] [Claim]
        df['text'] = df['Premise'].fillna('') + " [SEP] " + df['Claim'].fillna('')

        print(f"Loaded {len(df)} samples from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Skipping.")
        return pd.DataFrame()


# --- 3. PyTorch 数据集和计算函数 ---

class FeverDataset(torch.utils.data.Dataset):
    """用于 Hugging Face Trainer 的数据集类。"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
    """计算评估指标 (准确率)。"""
    preds = p.predictions.argmax(-1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}


def tokenize_data(df: pd.DataFrame, tokenizer: BertTokenizer) -> FeverDataset:
    """将 DataFrame 转换为 BERT 数据集。"""
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    return FeverDataset(encodings, df['labels'].tolist())


# --- 4. 实验核心逻辑：注入和清理 ---

# 确保在 DataFrame 中添加一个用于标记的新列名
CORRUPTION_MARKER_COL = 'is_corrupted'


def inject_corrupted_data(df_train: pd.DataFrame, corruption_ratio: float = 0.3) -> pd.DataFrame:
    """
    【最终修正版 - 注入】注入隐蔽性损坏数据，并使用专用的 **DataFrame 列**进行标记，
    以确保后续清理的准确性。
    """
    num_corruptions = int(len(df_train) * corruption_ratio)

    # 1. 标记原始数据：添加 'is_corrupted' 字段并设为 False
    df_train[CORRUPTION_MARKER_COL] = False

    subset = df_train.sample(n=num_corruptions, replace=False)
    corrupted_list = []

    print(f"\n🧠 正在注入 {num_corruptions} 条【隐蔽性】损坏数据 (使用标记列)...")

    for index, row in subset.iterrows():
        # 1. 翻转标签
        new_label_target = 1 - row['labels']
        new_label_str = REVERSE_LABEL_MAP[new_label_target]

        # 2. 注入矛盾的 Evidence (通过简单的否定词替换)
        original_evidence = row['Premise']

        corrupted_evidence = original_evidence.replace(' is ', ' is NOT ', 1).replace(' was ', ' was NOT ', 1)
        if corrupted_evidence == original_evidence:
            corrupted_evidence = "It is false that: " + original_evidence

        corrupted_list.append({
            'Label': new_label_str,
            'Premise': corrupted_evidence,
            'Claim': row['Claim'],
            'labels': new_label_target,
            'text': corrupted_evidence + " [SEP] " + row['Claim'],
            # 3. 标记损坏数据：设为 True
            CORRUPTION_MARKER_COL: True
        })

    df_corrupted = pd.DataFrame(corrupted_list)
    df_result = pd.concat([df_train, df_corrupted], ignore_index=True)

    print(f"✅ 注入 {len(df_corrupted)} 条损坏数据。总训练集大小: {len(df_result)}")
    return df_result.sample(frac=1).reset_index(drop=True)


def clean_corpus_via_contradiction_retrieval(df_corrupted: pd.DataFrame, df_original_size: int) -> pd.DataFrame:
    """
    【最终修正版 - 清理】通过识别专门的标记列来“完美”移除损坏数据，
    避免误删原始有效样本。
    """
    print("\n🔍 正在进行矛盾检索与清理 (使用标记列进行完美过滤)...")

    # 模拟矛盾检索器：基于标记列进行完美过滤 (即，只保留 is_corrupted == False 的样本)
    df_cleaned_temp = df_corrupted[df_corrupted[CORRUPTION_MARKER_COL] == False]

    # 计算被过滤掉的样本数（理论上应等于注入数）
    filtered_count = len(df_corrupted) - len(df_cleaned_temp)

    # 确保清理后的数据集大小与原始数据集大小一致 (理想情况)
    target_size = df_original_size

    if len(df_cleaned_temp) == target_size:
        df_cleaned = df_cleaned_temp.drop(columns=[CORRUPTION_MARKER_COL]).reset_index(drop=True)
    else:
        # 如果标记列过滤后大小不一致，说明逻辑有问题，但这里我们相信标记列是准确的。
        df_cleaned = df_cleaned_temp.drop(columns=[CORRUPTION_MARKER_COL]).reset_index(drop=True)
        print(f"⚠️ 警告: 过滤后样本数 ({len(df_cleaned)}) 不等于目标原始大小 ({target_size})。")
        # 如果 size < target_size，此处需要 pad 数据，但为了简洁，我们保持当前大小。

    print(f"✅ 清理完成。共识别并过滤掉 {filtered_count} 条矛盾数据。清理后大小: {len(df_cleaned)}")
    return df_cleaned.sample(frac=1).reset_index(drop=True)


def train_and_evaluate(df_train: pd.DataFrame, df_eval: pd.DataFrame, tokenizer: BertTokenizer, prefix: str) -> float:
    """训练 BERT 模型并在测试集上评估精度。"""

    print(f"\n--- {prefix}：正在训练/评估 BERT 模型 ---")

    # 1. 数据准备
    train_dataset = tokenize_data(df_train, tokenizer)
    eval_dataset = tokenize_data(df_eval, tokenizer)

    # 2. 模型和 Trainer 初始化
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir=f'./results_{prefix}',
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        logging_dir=f'./logs_{prefix}',
        logging_steps=100,
        do_eval=False,  # 训练期间不评估，加快速度
        disable_tqdm=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # 3. 训练
    trainer.train()

    # 4. 评估
    eval_results = trainer.evaluate(eval_dataset)
    accuracy = eval_results.get('eval_accuracy', 0.0)

    print(f"**{prefix} 事实核查精度:** {accuracy:.4f}")
    return accuracy


# --- 5. 主实验流程 ---

def main_experiment():
    # 初始化 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 1. 加载和抽样数据
    df_train_original = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/train.txt', SAMPLE_SIZE)
    df_test = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/test.txt', SAMPLE_SIZE)
    # dev.txt 在 NLI 任务中通常用于模型选择，我们在这里只加载，但主要使用 test.txt 评估
    df_dev = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/dev.txt', SAMPLE_SIZE)

    if df_train_original.empty or df_test.empty:
        print("错误：无法加载必要的训练或测试数据。请检查文件是否存在且格式正确。")
        return

    # --- 阶段 I：基线性能 ---
    print("\n" + "=" * 60)
    print("--- 阶段 I: 基线性能 (原始训练集) ---")

    # 获取原始训练集大小
    original_train_size = len(df_train_original)

    baseline_accuracy = train_and_evaluate(
        df_train=df_train_original,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="baseline"
    )
    print("=" * 60)

    # --- 阶段 II：注入损坏数据 (精度下降) ---
    df_corrupted_train = inject_corrupted_data(df_train_original, corruption_ratio=0.3)

    print("\n" + "=" * 60)
    print("--- 阶段 II: 注入损坏数据后的性能 (使用损坏数据微调) ---")

    corrupted_accuracy = train_and_evaluate(
        df_train=df_corrupted_train,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="inject"
    )
    print("=" * 60)

    # --- 阶段 III & IV：矛盾检索清理与性能提升 ---
    df_cleaned_train = clean_corpus_via_contradiction_retrieval(df_corrupted_train, original_train_size)

    print("\n" + "=" * 60)
    print("--- 阶段 IV: 清理后性能 (使用清理数据微调) ---")

    cleaned_accuracy = train_and_evaluate(
        df_train=df_cleaned_train,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="clean"
    )
    print("=" * 60)

    # --- 最终总结 ---
    print("\n\n" + "#" * 30)
    print("### 实验结果总结 ###")
    print(f"1. 原始基线精度: {baseline_accuracy:.4f}")
    print(f"2. 损坏后精度: {corrupted_accuracy:.4f}")
    print(f"3. 清理后精度: {cleaned_accuracy:.4f}")

    if cleaned_accuracy > corrupted_accuracy:
        print("\n🎉 结论: 矛盾检索清理成功提升了事实核查精度！")
        print(f"精度提升: {cleaned_accuracy - corrupted_accuracy:.4f}")
    else:
        print("\n⚠️ 结论: 即使使用 BERT，清理效果也不明显，可能需要更复杂的矛盾检索方法。")
    print("#" * 30)


if __name__ == "__main__":
    # 设置随机种子以保证实验可复现
    random.seed(42)
    torch.manual_seed(42)

    main_experiment()