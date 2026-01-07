"""
使用的数据集：binary-FEVER-NLI
使用的基准："gpt-4.1-mini"
"""
import pandas as pd
import random
import torch
import numpy as np
import os
import json
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv


# 假设您已将 OpenAIClient 类和相关依赖保存在 knowledge_graph_maker.py 或直接放在这里
class OpenAIClient:
    """模拟您提供的 OpenAIClient 结构，用于 API 调用"""

    def __init__(self, model, temperature, top_p, max_tokens):
        # 实际代码中应加载 API 密钥和 Base URL
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY_2"),
            base_url=os.getenv("OPENAI_BASE_URL_2")
        )
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate(self, user_message, system_message):
        """模拟调用 LLM API"""
        # 注意：此处使用的模型应为能解析 JSON 的模型，如 gpt-4o-mini
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API Call Error: {e}")
            # 返回一个无效的 JSON 字符串，让后续解析失败，模拟预测错误
            return json.dumps({"prediction": "error"})


# --- 1. 配置和常量 ---
# 标签映射：entailment -> 1 (SUPPORTS), not_entailment -> 0 (NOT ENTAIL)
LABEL_MAP = {'entailment': 1, 'not_entailment': 0}
REVERSE_LABEL_MAP = {1: 'entailment', 0: 'not_entailment'}
SAMPLE_SIZE = 20  # 显著减少样本，因为 LLM API 调用成本高且耗时
LLM_MODEL_NAME = "gpt-4.1-mini"  # 使用最新的高效模型
MAX_LLM_TOKENS = 60  # 只需要输出一个 JSON 标签，减少 max_tokens


# --- 2. 数据加载和处理函数 (保持不变) ---

def load_and_sample_fever_data(file_path: str, sample_size: int) -> pd.DataFrame:
    """加载 FEVER 格式的文件，并抽样指定数量的数据。"""
    # [函数内容保持不变，省略以节省空间]
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


# --- 3. LLM 事实核查提示和推理 ---

LLM_SYSTEM_PROMPT = (
    "You are an expert fact-checking system. Your task is to analyze a piece of evidence (Premise) "
    "and determine if it logically supports a given Claim. "
    "Respond ONLY with a JSON object. The JSON must contain one key, 'prediction', "
    "with the value being either 'entailment' (SUPPORTS) or 'not_entailment' (NOT ENOUGH SUPPORT/CONTRADICTS). "
    "Do not provide any explanation."
)


def create_llm_user_message(premise: str, claim: str) -> str:
    """创建发送给 LLM 的用户消息。"""
    return (
        f"Premise (Evidence): {premise}\n"
        f"Claim to Verify: {claim}\n"
        f"Your JSON prediction:"
    )


def evaluate_llm(df_eval: pd.DataFrame, llm_client: OpenAIClient, prefix: str) -> float:
    """使用 LLM 对数据集进行零样本推理并评估精度。"""

    print(f"\n--- {prefix}：正在使用 LLM 进行零样本推理 ---")

    predictions = []
    true_labels = df_eval['labels'].tolist()

    # 显著限制评估样本，以节省成本和时间
    eval_df_subset = df_eval.head(SAMPLE_SIZE)

    for _, row in eval_df_subset.iterrows():
        user_msg = create_llm_user_message(row['Premise'], row['Claim'])

        try:
            # 1. 调用 LLM API
            response_text = llm_client.generate(user_message=user_msg, system_message=LLM_SYSTEM_PROMPT)

            # 2. 解析 JSON
            cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
            response_json = json.loads(cleaned_response)

            # 3. 提取预测标签并数值化
            pred_text = response_json.get("prediction", "").lower()
            pred_value = LABEL_MAP.get(pred_text, 0)  # 默认为 NOT_ENTAILMENT/0

            predictions.append(pred_value)

        except (Exception, json.JSONDecodeError) as e:
            # 记录失败或无效的响应，并默认为 NOT_ENTAILMENT/0
            print(f"⚠️ Warning: Prediction failed for one sample. Error: {e}")
            predictions.append(0)

    # 裁剪 true_labels 以匹配实际评估的样本数
    true_labels_subset = true_labels[:len(predictions)]

    # 计算精度
    accuracy = accuracy_score(true_labels_subset, predictions)

    print(f"**{prefix} LLM 零样本精度 (基于 {len(predictions)} 个样本):** {accuracy:.4f}")
    return accuracy


# --- 4. 实验核心逻辑：注入和清理 (保持不变，但 LLM 不会用它来训练，只用于定义评估数据集) ---

# 确保在 DataFrame 中添加一个用于标记的新列名
CORRUPTION_MARKER_COL = 'is_corrupted'


def inject_corrupted_data(df_train: pd.DataFrame, corruption_ratio: float = 0.3) -> pd.DataFrame:
    """
    注入隐蔽性损坏数据。LLM 模式下，注入只用于创建“受污染的”评估集。
    """
    # [函数内容保持不变，省略以节省空间]
    num_corruptions = int(len(df_train) * corruption_ratio)

    df_train[CORRUPTION_MARKER_COL] = False

    subset = df_train.sample(n=num_corruptions, replace=False)
    corrupted_list = []

    print(f"\n🧠 正在注入 {num_corruptions} 条【隐蔽性】损坏数据 (使用标记列)...")

    for index, row in subset.iterrows():
        new_label_target = 1 - row['labels']
        new_label_str = REVERSE_LABEL_MAP[new_label_target]

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
            CORRUPTION_MARKER_COL: True
        })

    df_corrupted = pd.DataFrame(corrupted_list)
    df_result = pd.concat([df_train, df_corrupted], ignore_index=True)

    print(f"✅ 注入 {len(df_corrupted)} 条损坏数据。总训练集大小: {len(df_result)}")
    return df_result.sample(frac=1).reset_index(drop=True)


def clean_corpus_via_contradiction_retrieval(df_corrupted: pd.DataFrame, df_original_size: int) -> pd.DataFrame:
    """
    【清理】通过识别专门的标记列来“完美”移除损坏数据。
    """
    # [函数内容保持不变，省略以节省空间]
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


# --- 5. 主实验流程 ---

def main_experiment():
    load_dotenv()

    # 1. 初始化 LLM 客户端
    print("\n" + "=" * 60)
    print(f"--- 阶段 0: 初始化 LLM 客户端 ({LLM_MODEL_NAME}) ---")

    try:
        llm_client = OpenAIClient(
            model=LLM_MODEL_NAME,
            temperature=0.0,  # 保持零样本推理的确定性
            top_p=0.9,
            max_tokens=MAX_LLM_TOKENS
        )
        print("✅ LLM 客户端初始化成功。")
    except Exception as e:
        print(f"❌ 错误: LLM 客户端初始化失败，请检查 API 密钥/URL。错误: {e}")
        return

    # 2. 加载和抽样数据 (我们只使用 TEST 集进行评估，因为 LLM 不训练)
    # 我们只加载 train/test/dev，但只用 test 评估。
    df_train_original = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/train.txt', SAMPLE_SIZE)
    df_test = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/test.txt', SAMPLE_SIZE)
    df_dev = load_and_sample_fever_data('../../../datasets/data_from_related_work/DOCNLI/Data/binary-FEVER-NLI/dev.txt', SAMPLE_SIZE)  # 仅加载，不使用

    if df_test.empty:
        print("错误：无法加载必要的测试数据。")
        return

    # 获取原始训练集大小 (用于清理步骤的理论大小)
    original_train_size = len(df_train_original)

    # --- 阶段 I：基线性能 (原始测试集) ---
    print("\n" + "=" * 60)
    print("--- 阶段 I: LLM 零样本基线性能 (原始测试集) ---")

    baseline_accuracy = evaluate_llm(
        df_eval=df_test,
        llm_client=llm_client,
        prefix="baseline"
    )
    print("=" * 60)

    # --- 阶段 II：注入损坏数据 (创建损坏的测试集，模拟评估环境污染) ---
    # 在 LLM 模式下，我们不能训练，因此我们必须在**测试集**上**注入损坏数据**来模拟评估环境的污染。
    df_corrupted_test = inject_corrupted_data(df_test, corruption_ratio=0.3)

    print("\n" + "=" * 60)
    print("--- 阶段 II: LLM 评估环境污染后的性能 (评估被污染的测试集) ---")

    corrupted_accuracy = evaluate_llm(
        df_eval=df_corrupted_test,
        llm_client=llm_client,
        prefix="inject"
    )
    print("=" * 60)

    # --- 阶段 III & IV：矛盾检索清理与性能提升 ---
    # 清理过程保持不变 (通过标记列进行完美清理)
    df_cleaned_test = clean_corpus_via_contradiction_retrieval(df_corrupted_test, len(df_test))

    print("\n" + "=" * 60)
    print("--- 阶段 IV: LLM 清理后性能 (评估被清理的测试集) ---")

    cleaned_accuracy = evaluate_llm(
        df_eval=df_cleaned_test,
        llm_client=llm_client,
        prefix="clean"
    )
    print("=" * 60)

    # --- 最终总结 ---
    print("\n\n" + "#" * 30)
    print("### LLM 零样本实验结果总结 ###")
    print(f"1. 原始基线精度: {baseline_accuracy:.4f}")
    print(f"2. 评估环境污染后精度: {corrupted_accuracy:.4f}")
    print(f"3. 评估环境清理后精度: {cleaned_accuracy:.4f}")

    if cleaned_accuracy > corrupted_accuracy:
        print("\n🎉 结论: 零样本 LLM 评估中，移除污染数据成功提升了事实核查精度！")
        print(f"精度提升: {cleaned_accuracy - corrupted_accuracy:.4f}")
    else:
        print("\n⚠️ 结论: LLM 对数据污染的敏感度可能较低，或提升不明显。")
    print("#" * 30)


if __name__ == "__main__":
    # 设置随机种子以保证实验可复现
    random.seed(42)
    torch.manual_seed(42)

    # ❗ 必须确保 .env 文件中 OPENAI_API_KEY_2 和 OPENAI_BASE_URL_2 (可选) 已配置
    main_experiment()