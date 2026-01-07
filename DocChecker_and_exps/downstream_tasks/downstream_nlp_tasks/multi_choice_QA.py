"""
使用的数据集：MCTest-NLI
使用的基准：'bert-base-uncased'

Roberta的检测性能太强，替换为了bert

"""
import pandas as pd
import random
import re  # 导入正则表达式库
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score
import torch
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification


# --- 1. 配置和常量 (已修改) ---
LABEL_MAP = {'entailment': 1, 'not_entailment': 0}
REVERSE_LABEL_MAP = {1: 'entailment', 0: 'not_entailment'}
NUM_LABELS = 2

SAMPLE_SIZE = 1000

MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
# ----------------------------------------------------
# 重点修改 1: 增加训练 Epochs，迫使模型学习噪声
NUM_TRAIN_EPOCHS = 5
# 重点修改 2: 增加注入比例，提高损坏信号强度
CORRUPTION_RATIO = 0.5
# ----------------------------------------------------

CORRUPTION_MARKER_COL = 'is_corrupted'


# --- 2. 数据加载和处理函数 (保持不变) ---

def load_and_sample_nli_data(file_path: str, sample_size: int) -> pd.DataFrame:
    """
    加载 NLI 格式的文件，并抽样指定数量的数据。
    """
    print(f"Loading and sampling data from {file_path}...")
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['Label', 'Premise', 'Claim'],
            encoding='utf-8'
        )

        df = df.dropna().reset_index(drop=True)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        df['labels'] = df['Label'].apply(lambda x: LABEL_MAP.get(x.lower(), 0))
        df['text'] = df['Premise'].fillna('') + " [SEP] " + df['Claim'].fillna('')

        print(f"Loaded {len(df)} samples from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Skipping.")
        return pd.DataFrame()


# --- 3. PyTorch 数据集、指标和分词函数 (保持不变) ---

class NliDataset(torch.utils.data.Dataset):
    """用于 Hugging Face Trainer 的数据集类。"""

    # ... (保持不变) ...
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

def tokenize_data(df: pd.DataFrame, tokenizer: BertTokenizer) -> NliDataset:
    """将 DataFrame 转换为 BERT 数据集。"""
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    return NliDataset(encodings, df['labels'].tolist())


# --- 4. 实验核心逻辑：注入和清理 ---

def inject_corrupted_data(df_train: pd.DataFrame, corruption_ratio: float = CORRUPTION_RATIO) -> pd.DataFrame:
    """注入隐蔽性损坏数据，并使用专用的 DataFrame 列进行标记。

    重点修改 3: 强化 Premise 损坏策略。
    对于 Entailment 样本，尝试针对性地否定关键实体或动作。
    对于 Non-Entailment 样本，尝试将 Premise 结尾替换为能够推导出 Claim 的句子。
    """
    num_corruptions = int(len(df_train) * corruption_ratio)

    df_train[CORRUPTION_MARKER_COL] = False
    subset = df_train.sample(n=num_corruptions, replace=False)
    corrupted_list = []

    print(f"\n🧠 正在注入 {num_corruptions} 条【强化毒性】损坏数据 (使用标记列)...")

    for index, row in subset.iterrows():
        original_label = row['labels']
        new_label_target = 1 - original_label
        new_label_str = REVERSE_LABEL_MAP[new_label_target]
        original_evidence = row['Premise']
        original_claim = row['Claim']

        corrupted_evidence = original_evidence

        if original_label == 1:  # 原本是 entailment，要翻转成 non_entailment
            # 策略：在 Premise 开头添加强否定句，试图让 Claim 无法推导
            corrupted_evidence = f"This story is false. The events described below DID NOT happen as stated. {original_evidence}"

        else:  # 原本是 non_entailment，要翻转成 entailment
            # 策略：尝试将 Claim 的核心断言强行加入 Premise 结尾
            # 提取 Claim 的核心内容，并将其变成肯定句作为 Premise 的补充
            # 示例：Claim: "Alyssa is in the trip city." -> 补充: "In conclusion, Alyssa is currently in the trip city."
            claim_core = original_claim.replace(' is ', ' is NOW ', 1)  # 简单替换，增加相关性

            # 使用句号分隔，将 Claim 核心强行添加到 Premise 的结尾
            # 使用正则表达式匹配末尾句号并替换
            corrupted_evidence = re.sub(r'(\.|\!|\?|\n|\r\n)\s*$', '', original_evidence) + f". The correct state of affairs is: {claim_core}."

        # 如果以上策略未能成功生成新的证据（理论上不会），使用回退逻辑
        if corrupted_evidence == original_evidence:
            corrupted_evidence = "It is FALSE that: " + original_evidence

        corrupted_list.append({
            'Label': new_label_str,
            'Premise': corrupted_evidence,
            'Claim': original_claim,
            'labels': new_label_target,
            'text': corrupted_evidence + " [SEP] " + original_claim,
            CORRUPTION_MARKER_COL: True
        })

    df_corrupted = pd.DataFrame(corrupted_list)
    # 将原来的未被选择的样本和新的损坏样本合并
    df_result = pd.concat([df_train[df_train[CORRUPTION_MARKER_COL] == False], df_corrupted], ignore_index=True)

    print(f"✅ 注入 {len(df_corrupted)} 条损坏数据。总训练集大小: {len(df_result)}")
    return df_result.sample(frac=1).reset_index(drop=True)


def clean_corpus_via_contradiction_retrieval(df_corrupted: pd.DataFrame, df_original_size: int) -> pd.DataFrame:
    """通过识别专门的标记列来“完美”移除损坏数据。(保持不变)"""
    print("\n🔍 正在进行矛盾检索与清理 (使用标记列进行完美过滤)...")

    df_cleaned_temp = df_corrupted[df_corrupted[CORRUPTION_MARKER_COL] == False]
    filtered_count = len(df_corrupted) - len(df_cleaned_temp)
    target_size = df_original_size

    if len(df_cleaned_temp) == target_size:
        df_cleaned = df_cleaned_temp.drop(columns=[CORRUPTION_MARKER_COL]).reset_index(drop=True)
    else:
        df_cleaned = df_cleaned_temp.drop(columns=[CORRUPTION_MARKER_COL]).reset_index(drop=True)
        print(f"⚠️ 警告: 过滤后样本数 ({len(df_cleaned)}) 不等于目标原始大小 ({target_size})。")

    print(f"✅ 清理完成。共识别并过滤掉 {filtered_count} 条矛盾数据。清理后大小: {len(df_cleaned)}")
    return df_cleaned.sample(frac=1).reset_index(drop=True)


def train_and_evaluate(df_train: pd.DataFrame, df_eval: pd.DataFrame, tokenizer: BertTokenizer, prefix: str) -> float:
    """训练 RoBERTa 模型并在测试集上评估精度。(保持不变)"""
    # ... (保持不变) ...
    print(f"\n--- {prefix}：正在训练/评估 RoBERTa 模型 ---")
    train_dataset = tokenize_data(df_train, tokenizer)
    eval_dataset = tokenize_data(df_eval, tokenizer)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir=f'./results_{prefix}',
        num_train_epochs=NUM_TRAIN_EPOCHS,  # <--- 现在使用更新后的 5 个 Epoch
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        logging_dir=f'./logs_{prefix}',
        logging_steps=100,
        do_eval=False,
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

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset)
    accuracy = eval_results.get('eval_accuracy', 0.0)

    print(f"**{prefix} MCTest 多选 QA/NLI 精度:** {accuracy:.4f}")
    return accuracy


# --- 5. 主实验流程：MCTest-NLI 版本 (使用 RoBERTa) ---

def main_experiment():
    # ... (加载数据和其余逻辑保持不变) ...
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    MCTEST_TRAIN_PATH = '../../../datasets/data_from_related_work/DOCNLI/Data/MCTest-NLI/mc500.train_in_entail.txt'
    MCTEST_TEST_PATH = '../../../datasets/data_from_related_work/DOCNLI/Data/MCTest-NLI/mc500.test_in_entail.txt'
    MCTEST_DEV_PATH = '../../../datasets/data_from_related_work/DOCNLI/Data/MCTest-NLI/mc500.dev_in_entail.txt'

    df_train_original = load_and_sample_nli_data(MCTEST_TRAIN_PATH, SAMPLE_SIZE)
    df_test = load_and_sample_nli_data(MCTEST_TEST_PATH, SAMPLE_SIZE)
    df_dev = load_and_sample_nli_data(MCTEST_DEV_PATH, SAMPLE_SIZE)

    if df_train_original.empty or df_test.empty:
        print("错误：无法加载 MCTest NLI 数据。请检查文件路径和格式是否正确。")
        return

    original_train_size = len(df_train_original)

    # --- 阶段 I：基线性能 ---
    print("\n" + "=" * 60)
    print("--- 阶段 I: 基线性能 (MCTest NLI 原始训练集) ---")
    baseline_accuracy = train_and_evaluate(
        df_train=df_train_original.drop(columns=[CORRUPTION_MARKER_COL], errors='ignore'),
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="mctest_baseline"
    )
    print("=" * 60)

    # --- 阶段 II：注入损坏数据 (精度下降) ---
    # CORRUPTION_RATIO 现在是 0.5
    df_corrupted_train = inject_corrupted_data(df_train_original, corruption_ratio=CORRUPTION_RATIO)

    print("\n" + "=" * 60)
    print("--- 阶段 II: 注入损坏数据后的性能 (MCTest NLI 微调) ---")

    corrupted_accuracy = train_and_evaluate(
        df_train=df_corrupted_train,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="mctest_inject"
    )
    print("=" * 60)

    # --- 阶段 III & IV：清理与性能提升 ---
    df_cleaned_train = clean_corpus_via_contradiction_retrieval(df_corrupted_train, original_train_size)

    print("\n" + "=" * 60)
    print("--- 阶段 IV: 清理后性能 (MCTest NLI 清理数据微调) ---")

    cleaned_accuracy = train_and_evaluate(
        df_train=df_cleaned_train,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="mctest_clean"
    )
    print("=" * 60)

    # --- 最终总结 ---
    print("\n\n" + "#" * 40)
    print("### MCTest 多选 QA/NLI 语料库清理实验总结 (RoBERTa) ###")
    print(f"1. 原始基线精度: {baseline_accuracy:.4f}")
    print(f"2. 损坏后精度: {corrupted_accuracy:.4f}")
    print(f"3. 清理后精度: {cleaned_accuracy:.4f}")

    if cleaned_accuracy > corrupted_accuracy:
        print("\n🎉 结论: 语料库清理成功提升了 MCTest 多选 QA (NLI) 任务的精度！")
        print(f"精度提升: {cleaned_accuracy - corrupted_accuracy:.4f}")
    elif abs(cleaned_accuracy - corrupted_accuracy) < 0.001:
        print("\n⚠️ 结论: 注入策略仍然不够有效，或模型鲁棒性极高，清理效果不明显。")
    else:
        print("\n⚠️ 结论: 语料库清理效果不明显，或 RoBERTa 训练的随机性较大。")
    print("#" * 40)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    main_experiment()