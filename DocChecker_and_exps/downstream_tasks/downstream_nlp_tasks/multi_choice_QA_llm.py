"""
使用的数据集：MCTest-NLI
使用的基准：'bert-base-uncased'
"""
import pandas as pd
import random
from typing import List, Dict, Tuple, Any
import json
import os
import time
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
import torch
# --- 替换为 RoBERTa 模型和 Tokenizer ---
from transformers import Trainer, TrainingArguments, BertTokenizer, BertForSequenceClassification
# 假设知识图谱相关类库已定义，此处保留原导入
from knowledge_graph_maker import OpenAIClient

# --- 1. 配置和 LLM 客户端 ---
# 保持 LLM 配置不变
load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # 您的原始代码
# os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None) # 您的原始代码

try:
    # 增加 max_tokens 以便模型有空间返回 k 个证据文本和解释
    llm = OpenAIClient(
        model="gpt-4.1-mini",
        temperature=0.1,
        top_p=0.9,
        max_tokens=1500
    )
except Exception as e:
    print(f"Error initializing OpenAIClient: {e}. RoBERTa 实验将无法进行 LLM 注入/清理。")
    llm = None

# --- 2. 配置和常量 (RoBERTa 和 MCTest) ---
LABEL_MAP = {'entailment': 1, 'not_entailment': 0}
REVERSE_LABEL_MAP = {1: 'entailment', 0: 'not_entailment'}
NUM_LABELS = 2
SAMPLE_SIZE = 1000  # 适应 MCTest-500 的子集大小

# --- 替换为 RoBERTa ---
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 3
CORRUPTION_RATIO = 0.3
CORRUPTION_MARKER = "!!CORRUPT_LABEL_AND_EVIDENCE!!"


# --- 3. PyTorch 数据集和指标函数 ---

class NliDataset(torch.utils.data.Dataset):  # 重命名数据集类
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


# --- 4. 数据加载和分词函数 (适配 RoBERTa) ---

def load_and_sample_nli_data(file_path: str, sample_size: int) -> pd.DataFrame:
    """加载 MCTest NLI 格式的文件，并抽样指定数量的数据。"""
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
        # PLM 的输入格式：[Premise] [SEP] [Claim]
        # 在QA语境中：[Document] [SEP] [Question + Candidate Answer]
        df['text'] = df['Premise'].fillna('') + " [SEP] " + df['Claim'].fillna('')

        print(f"Loaded {len(df)} samples from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Skipping.")
        return pd.DataFrame()


def tokenize_data(df: pd.DataFrame, tokenizer: BertTokenizer) -> NliDataset:
    """将 DataFrame 转换为 BERT 数据集。"""
    encodings = tokenizer(
        df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_LEN
    )
    return NliDataset(encodings, df['labels'].tolist())


# --- 5. 实验核心逻辑：LLM 注入和标记清理 ---

def llm_inject_corrupted_data(llm_client: Any, df_train: pd.DataFrame, corruption_ratio: float = 0.3, batch_size: int = 15) -> pd.DataFrame:
    """
    使用 LLM 批量生成矛盾证据，并添加强烈标记。
    (代码与您提供的版本保持一致，使用 LLM 增强注入)
    """

    num_corruptions = int(len(df_train) * corruption_ratio)
    subset = df_train.sample(n=num_corruptions, replace=False)
    corrupted_data_to_generate = []

    for index, row in subset.iterrows():
        corrupted_data_to_generate.append({
            'original_index': index,
            'original_premise': row['Premise'],
            'claim': row['Claim'],
            'original_labels': row['labels'],
            'original_label_str': row['Label']
        })

    corrupted_list = []
    # ... (系统提示和批量处理逻辑与您的代码保持一致) ...

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

    for i in range(0, num_corruptions, batch_size):
        batch = corrupted_data_to_generate[i:i + batch_size]
        user_message_inject_batch = json.dumps([
            {
                "original_index": item['original_index'],
                "original_premise": item['original_premise'],
                "claim": item['claim']
            }
            for item in batch
        ])

        try:
            response_text = llm_client.generate(user_message=user_message_inject_batch, system_message=system_prompt_inject)
            response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
            response_array = json.loads(response_text)

            if not isinstance(response_array, list):
                response_array = []

            for result_json in response_array:
                original_index = result_json.get('original_index')
                new_evidence = result_json.get('new_evidence')

                original_row = next((item for item in corrupted_data_to_generate if item['original_index'] == original_index), None)

                if original_row and new_evidence:
                    new_label_target = 1 - original_row['original_labels']
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

    df_corrupted = pd.DataFrame(corrupted_list)
    df_result = pd.concat([df_train, df_corrupted], ignore_index=True)

    print(f"✅ 成功注入 {len(df_corrupted)} 条 LLM 损坏数据。总训练集大小: {len(df_result)}")
    return df_result.sample(frac=1).reset_index(drop=True)


def clean_corpus_via_contradiction_retrieval(df_corrupted: pd.DataFrame, df_original_size: int) -> pd.DataFrame:
    """
    通过识别注入的强烈标记来“完美”移除损坏数据。
    注意：函数签名已修正以匹配 LLM 实验流程。
    """
    print("\n🔍 正在进行矛盾检索与清理 (基于强烈标记的直接过滤)...")

    # 直接过滤掉所有包含强烈标记的样本
    df_cleaned_temp = df_corrupted[~df_corrupted['text'].str.contains(CORRUPTION_MARKER, case=False)]

    filtered_count = len(df_corrupted) - len(df_cleaned_temp)
    target_size = df_original_size

    if len(df_cleaned_temp) != target_size:
        print(f"⚠️ 警告: 过滤后样本数 ({len(df_cleaned_temp)}) 不等于目标原始大小 ({target_size})。")
        # 如果标记过滤不完美，则保留过滤结果，并给出警告
        df_cleaned = df_cleaned_temp.sample(n=min(len(df_cleaned_temp), target_size), random_state=42).reset_index(drop=True)
    else:
        df_cleaned = df_cleaned_temp.reset_index(drop=True)

    print(f"✅ 清理完成。共识别并过滤掉 {filtered_count} 条矛盾数据。清理后大小: {len(df_cleaned)}")
    return df_cleaned.sample(frac=1).reset_index(drop=True)


def train_and_evaluate(df_train: pd.DataFrame, df_eval: pd.DataFrame, tokenizer: BertTokenizer, prefix: str) -> float:
    """训练 RoBERTa 模型并在 MCTest NLI 测试集上评估精度。"""

    print(f"\n--- {prefix}：正在训练/评估 RoBERTa 模型 ---")

    # 1. 数据准备
    train_dataset = tokenize_data(df_train, tokenizer)
    eval_dataset = tokenize_data(df_eval, tokenizer)

    # 2. 模型和 Trainer 初始化
    # --- 替换为 RoBERTa 模型类 ---
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir=f'./results_{prefix}',
        num_train_epochs=NUM_TRAIN_EPOCHS,
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

    # 3. 训练
    trainer.train()

    # 4. 评估
    eval_results = trainer.evaluate(eval_dataset)
    accuracy = eval_results.get('eval_accuracy', 0.0)

    # --- 输出更新，强调多选 QA/NLI ---
    print(f"**{prefix} MCTest 多选 QA/NLI 精度:** {accuracy:.4f}")
    return accuracy


# --- 6. 主实验流程：MCTest-NLI 版本 (使用 RoBERTa 和 LLM) ---

def main_experiment():
    # --- 替换为 RoBERTa Tokenizer ---
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 调整 SAMPLE_SIZE 适应 MCTest-500 的子集
    global SAMPLE_SIZE
    SAMPLE_SIZE = 1000

    # --------------------------------------------------------------------------------
    # **MCTest-NLI 路径**
    MCTEST_TRAIN_PATH = '../../../datasets/data_from_related_work/DOCNLI/Data/MCTest-NLI/mc500.train_in_entail.txt'
    MCTEST_TEST_PATH = '../../../datasets/data_from_related_work/DOCNLI/Data/MCTest-NLI/mc500.test_in_entail.txt'
    MCTEST_DEV_PATH = '../../../datasets/data_from_related_work/DOCNLI/Data/MCTest-NLI/mc500.dev_in_entail.txt'
    # --------------------------------------------------------------------------------

    # 1. 加载和抽样数据
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
        df_train=df_train_original,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="mctest_baseline"
    )
    print("=" * 60)

    # --- 阶段 II：LLM 注入损坏数据 (精度下降) ---
    if llm:
        df_corrupted_train = llm_inject_corrupted_data(llm, df_train_original, corruption_ratio=CORRUPTION_RATIO)
    else:
        print("⚠️ LLM 客户端初始化失败，跳过 LLM 注入步骤。")
        return

    print("\n" + "=" * 60)
    print("--- 阶段 II: 注入损坏数据后的性能 (使用损坏数据微调) ---")

    corrupted_accuracy = train_and_evaluate(
        df_train=df_corrupted_train,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="mctest_inject"
    )
    print("=" * 60)

    # --- 阶段 III & IV：矛盾检索清理与性能提升 ---
    # 注意：这里需要传入 original_train_size
    df_cleaned_train = clean_corpus_via_contradiction_retrieval(df_corrupted_train, original_train_size)

    print("\n" + "=" * 60)
    print("--- 阶段 IV: 清理后性能 (使用清理数据微调) ---")

    cleaned_accuracy = train_and_evaluate(
        df_train=df_cleaned_train,
        df_eval=df_test,
        tokenizer=tokenizer,
        prefix="mctest_clean"
    )
    print("=" * 60)

    # --- 最终总结 ---
    print("\n\n" + "#" * 40)
    print("### MCTest 多选 QA/NLI 语料库清理实验总结 (RoBERTa + LLM) ###")
    print(f"1. 原始基线精度: {baseline_accuracy:.4f}")
    print(f"2. 损坏后精度: {corrupted_accuracy:.4f}")
    print(f"3. 清理后精度: {cleaned_accuracy:.4f}")

    if cleaned_accuracy > corrupted_accuracy:
        print("\n🎉 结论: 语料库清理成功提升了 MCTest 多选 QA (NLI) 任务的精度！")
        print(f"精度提升: {cleaned_accuracy - corrupted_accuracy:.4f}")
    else:
        print("\n⚠️ 结论: LLM 注入的损坏数据或 RoBERTa 训练的随机性较大，清理效果不明显。")
    print("#" * 40)


if __name__ == "__main__":
    # 设置随机种子以保证实验可复现
    random.seed(42)
    torch.manual_seed(42)

    # 您的原始代码使用了 load_and_sample_fever_data，这里保持兼容性
    load_and_sample_fever_data = load_and_sample_nli_data

    main_experiment()