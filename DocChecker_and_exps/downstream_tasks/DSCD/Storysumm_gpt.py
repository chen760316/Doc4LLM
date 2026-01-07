# DSCD_StorySumm_GPT_Inference.py
"""
使用的数据集：StorySumm
使用的基准："gpt-4.1-mini"
"""
# 假设您已安装并配置好这些库
from knowledge_graph_maker import OpenAIClient
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.metrics import classification_report
import numpy as np
import json
import os
from dotenv import load_dotenv
from typing import List, Dict

# --- 1. 配置和 LLM 客户端 ---
load_dotenv()

# 请替换为您实际的 API 密钥和 Base URL 配置
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 初始化 LLM 客户端
llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=500  # 减少 token 数量，只需要简短的 JSON 响应
)

# --- 2. 专门用于 DSCD 的系统提示 ---
CONFLICT_SYSTEM_PROMPT = (
    "You are an expert in detecting contradictions within a single document.\n"
    "Your task is to analyze the following document (a summary) and determine if it contains any self-contradictory information, "
    "where two or more statements within the text are logically incompatible with each other.\n"
    "Respond with a single JSON object with two keys:\n"
    '{"is_contradictory": <boolean>, "reason": "<brief explanation of the conflict, or why none was found>"}\n'
    "Respond with only the JSON object—no explanations, no markdown, no extra text."
)


# --- 3. 数据加载和映射函数 (与您提供的代码相同) ---

# --- 1. 数据加载和映射函数 (修正版本) ---

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


# --- 4. GPT API 调用和解析函数 ---

def detect_conflict_with_gpt(llm_client: OpenAIClient, document_text: str, system_prompt: str) -> int:
    """使用 GPT 模型检测文档内冲突，并返回整数标签 (0 或 1)"""
    try:
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt)

        # 清理并解析 JSON 响应
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()

        response_json = json.loads(response_text)

        # 返回预测的标签 (1 for True/Contradictory, 0 for False/Non-Contradictory)
        return int(response_json.get("is_contradictory", False))

    except Exception as e:
        # 在 API 调用失败或 JSON 解析失败时，输出错误并返回默认值
        # 默认返回 0 (无矛盾)
        print(f"Error processing document (returning 0): {e}. Raw Response: {response_text[:100]}...")
        return 0


# --- 5. 主执行逻辑 ---

if __name__ == '__main__':
    # 限制 LLM 调用的样本数，因为 LLM 调用较慢且收费
    LIMIT = 20

    # 您的文件路径
    STORYSUMM_FILE_PATH = "../../../datasets/data_from_related_work/RRC-DSCD/data/raw_data/storysumm.json"

    print("🌍 正在加载 StorySumm 数据集并映射为 DSCD 任务...")
    raw_dataset = load_and_map_storysumm(STORYSUMM_FILE_PATH)

    # 划分数据集（仅使用测试集进行 zero-shot 评估）
    # 分离出 20% 作为测试集
    train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset_full = train_test_split['test']

    # 限制测试集大小
    test_dataset = test_dataset_full.select(range(min(LIMIT, len(test_dataset_full))))

    print(f"⚠️ 正在对限制后的 {len(test_dataset)} 条测试数据进行 GPT 零样本检测...")

    # 提取真实标签
    true_labels = test_dataset["label"]
    predicted_labels = []

    # --- 核心：循环调用 GPT API ---
    for i, example in enumerate(test_dataset):
        # 提取 document 字段
        document_text = example['document']

        print(f"Processing {i + 1}/{len(test_dataset)}: ID {example['original_story_id']}...")

        prediction = detect_conflict_with_gpt(llm, document_text, CONFLICT_SYSTEM_PROMPT)
        predicted_labels.append(prediction)

    # --- 评估结果 ---

    # 打印详细报告
    print("\n--- 📊 GPT 零样本冲突检测结果 ---")

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=['Non-Contradictory (0)', 'Contradictory (1)'],
        digits=4
    )
    print("详细分类报告:")
    print(report)