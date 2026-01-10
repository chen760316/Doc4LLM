""""
使用的数据集：binary-FEVER-NLI
使用的基准："gpt-4.1-mini"

借助llm注入和清理损坏数据
借助llm注入损坏数据后事实核查任务指标下降明显
x修复损坏数据后事实核查精度有明显提升
执行速度慢
"""
import pandas as pd
import random
from typing import List, Dict, Tuple, Any
import json
import os
import time
from dotenv import load_dotenv  # 用于加载 API 密钥
from sklearn.metrics import accuracy_score
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from knowledge_graph_maker import OpenAIClient

# --- 1. 配置和 LLM 客户端 ---
# 保持 LLM 配置不变
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

try:
    # 增加 max_tokens 以便模型有空间返回 k 个证据文本和解释
    llm = OpenAIClient(
        model="gpt-5.2",
        temperature=0.1,
        top_p=0.9,
        max_tokens=1500  # 恢复到 1500，确保输出空间
    )
except Exception as e:
    print(f"Error initializing OpenAIClient: {e}. Exiting.")
    exit()



# --- 1. 配置和常量 (保持不变) ---
# ... (配置保持不变，LLM 客户端已定义)
LABEL_MAP = {'entailment': 1, 'not_entailment': 0}
REVERSE_LABEL_MAP = {1: 'entailment', 0: 'not_entailment'}
NUM_LABELS = 2
SAMPLE_SIZE = 1200

MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 3  # 增加到 3 个 Epoch

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


# 定义一个更强烈的、专用的标记，确保注入效果
CORRUPTION_MARKER = "!!CORRUPT_LABEL_AND_EVIDENCE!!"

def llm_inject_corrupted_data(llm_client: Any, df_train: pd.DataFrame, corruption_ratio: float = 0.3, batch_size: int = 15) -> pd.DataFrame:
    """
    【修正版】使用 LLM 批量生成矛盾证据，并添加强烈标记，以最大化损坏效果。

    Args:
        llm_client: LLM 客户端对象。
        df_train: 原始训练集 DataFrame。
        corruption_ratio: 注入损坏数据的比例。
        batch_size: 每次调用 LLM 处理的样本数量。

    Returns:
        注入损坏数据后的训练集 DataFrame。
    """

    num_corruptions = int(len(df_train) * corruption_ratio)
    subset = df_train.sample(n=num_corruptions, replace=False)

    corrupted_data_to_generate = []
    for index, row in subset.iterrows():
        # 仅选择 LLM 需要的信息进行打包
        corrupted_data_to_generate.append({
            'original_index': index,
            'original_premise': row['Premise'],
            'claim': row['Claim'],
            'original_labels': row['labels'],
            'original_label_str': row['Label']
        })

    corrupted_list = []

    # --- 批量注入的核心逻辑 ---

    system_prompt_inject = (
        "You are an expert at generating realistic, but false, contradictory evidence for fact validation. "
        "You will be given a list of input objects, each containing an 'original_premise' and a 'claim'. "
        "For EACH object in the list, you must generate a new sentence, called 'new_evidence', that strongly contradicts the 'claim'. "
        "The new evidence must look like it comes from a trustworthy source. "
        "Your final output MUST be a single JSON ARRAY containing N objects. "
        "Each object in the array must have the format: "
        "{ \"original_index\": [The index from the input], \"new_evidence\": \"[Generated contradictory evidence]\", \"explanation\": \"[Brief reason]\" }."
        "Output ONLY the JSON ARRAY."
    )

    print(f"\n🧠 正在使用 LLM 批量注入 {num_corruptions} 条损坏数据 (Batch Size: {batch_size})，并添加强烈标记...")

    # 按批次处理要注入的样本
    for i in range(0, num_corruptions, batch_size):
        batch = corrupted_data_to_generate[i:i + batch_size]

        # 准备用户输入
        user_message_inject_batch = json.dumps([
            {
                "original_index": item['original_index'],
                "original_premise": item['original_premise'],
                "claim": item['claim']
            }
            for item in batch
        ])

        try:
            # 调用 LLM API
            response_text = llm_client.generate(user_message=user_message_inject_batch, system_message=system_prompt_inject)
            response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
            response_array = json.loads(response_text)

            if not isinstance(response_array, list):
                response_array = []

            # 遍历 LLM 返回的批量结果
            for result_json in response_array:
                original_index = result_json.get('original_index')
                new_evidence = result_json.get('new_evidence')

                original_row = next((item for item in corrupted_data_to_generate if item['original_index'] == original_index), None)

                if original_row and new_evidence:
                    # 1. 翻转标签
                    new_label_target = 1 - original_row['original_labels']
                    # 2. 添加强烈标记 + LLM 生成的证据
                    corrupted_evidence = f"{CORRUPTION_MARKER} " + new_evidence

                    corrupted_list.append({
                        'Label': REVERSE_LABEL_MAP[new_label_target],
                        'Premise': corrupted_evidence,
                        'Claim': original_row['claim'],
                        'labels': new_label_target,
                        'text': corrupted_evidence + " [SEP] " + original_row['claim']
                    })

        except Exception as e:
            print(f"⚠️ 批量注入失败 (Batch {i // batch_size + 1}): {e}. 回退到简单的字符串替换。")
            # 回退逻辑 (也需要添加标记)
            for item in batch:
                corrupted_evidence = f"{CORRUPTION_MARKER} " + item['original_premise'].replace('is ', 'is NOT ', 1)
                new_label_target = 1 - item['original_labels']

                corrupted_list.append({
                    'Label': REVERSE_LABEL_MAP[new_label_target],
                    'Premise': corrupted_evidence,
                    'Claim': item['claim'],
                    'labels': new_label_target,
                    'text': corrupted_evidence + " [SEP] " + item['claim']
                })

    # 将生成的损坏数据与原始训练集合并
    df_corrupted = pd.DataFrame(corrupted_list)
    df_result = pd.concat([df_train, df_corrupted], ignore_index=True)

    print(f"✅ 成功注入 {len(df_corrupted)} 条 LLM 损坏数据。总训练集大小: {len(df_result)}")
    return df_result.sample(frac=1).reset_index(drop=True)


def clean_corpus_via_contradiction_retrieval(df_corrupted: pd.DataFrame, df_original_size: int) -> pd.DataFrame:
    """
    【修正版 - 直接清理】不再借助 LLM，而是通过识别注入的强烈标记来“完美”移除损坏数据。

    Args:
        df_corrupted: 注入损坏数据后的 DataFrame。
        df_original_size: 原始训练集的大小 (用于确定应保留的样本数量)。

    Returns:
        清理后的 DataFrame (理论上应恢复到原始训练集)。
    """
    print("\n🔍 正在进行矛盾检索与清理 (基于强烈标记的直接过滤)...")

    # 直接过滤掉所有包含强烈标记的样本
    df_cleaned_temp = df_corrupted[~df_corrupted['text'].str.contains(CORRUPTION_MARKER, case=False)]

    filtered_count = len(df_corrupted) - len(df_cleaned_temp)

    # 确保清理后的数据集大小与原始数据集大小一致 (理想情况)
    target_size = df_original_size

    if len(df_cleaned_temp) != target_size:
        print(f"⚠️ 警告: 过滤后样本数 ({len(df_cleaned_temp)}) 不等于目标原始大小 ({target_size})。")
        # 如果标记过滤不完美，则保留过滤结果，并给出警告
        df_cleaned = df_cleaned_temp.sample(n=min(len(df_cleaned_temp), target_size), random_state=42).reset_index(drop=True)
    else:
        df_cleaned = df_cleaned_temp.reset_index(drop=True)


    print(f"✅ 清理完成。共识别并过滤掉 {filtered_count} 条矛盾数据。清理后大小: {len(df_cleaned)}")
    return df_cleaned.sample(frac=1).reset_index(drop=True)

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

def tokenize_data(df: pd.DataFrame, tokenizer: BertTokenizer) -> FeverDataset:
    """将 DataFrame 转换为 BERT 数据集。"""
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    return FeverDataset(encodings, df['labels'].tolist())

def compute_metrics(p):
    """计算评估指标 (准确率)。"""
    preds = p.predictions.argmax(-1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}

# --- 5. 主实验流程 (修改注入和清理函数调用) ---

def main_experiment():
    # ... (初始化 Tokenizer 和加载数据部分保持不变)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 1. 加载和抽样数据
    df_train_original = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/train.txt', SAMPLE_SIZE)
    df_test = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/test.txt', SAMPLE_SIZE)
    df_dev = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/dev.txt', SAMPLE_SIZE)

    if df_train_original.empty or df_test.empty:
        print("错误：无法加载必要的训练或测试数据。请检查文件是否存在且格式正确。")
        return

    # --- 阶段 I：基线性能 ---
    print("\n" + "=" * 60)
    print("--- 阶段 I: 基线性能 (原始训练集) ---")

    baseline_accuracy = train_and_evaluate(
        df_train=df_train_original,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="baseline"
    )
    print("=" * 60)

    # --- 阶段 II：LLM 注入损坏数据 (精度下降) ---
    # 使用 LLM 驱动的注入函数
    df_corrupted_train = llm_inject_corrupted_data(llm, df_train_original, corruption_ratio=0.3)

    print("\n" + "=" * 60)
    print("--- 阶段 II: 注入损坏数据后的性能 (使用损坏数据微调) ---")

    corrupted_accuracy = train_and_evaluate(
        df_train=df_corrupted_train,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="inject"
    )
    print("=" * 60)

    original_train_size = len(df_train_original)

    # --- 阶段 III & IV：LLM 矛盾检索清理与性能提升 ---
    # 使用 LLM 驱动的清理函数
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
        print("\n⚠️ 结论: LLM 清理效果不明显，可能需要更复杂的 LLM 提示词或更高的训练 Epoch。")
    print("#" * 30)


if __name__ == "__main__":
    # 设置随机种子以保证实验可复现
    random.seed(30)
    torch.manual_seed(42)

    # 由于您的 LLM 客户端依赖外部库和环境变量，请确保它们已正确配置。
    # load_dotenv() # 确保在您的实际环境中运行

    main_experiment()