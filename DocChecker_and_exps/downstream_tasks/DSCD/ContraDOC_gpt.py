# DSCD_GPT_Inference.py
"""
使用的数据集：ContraDOC
使用的基准："gpt-4.1-mini"
"""
from knowledge_graph_maker import OpenAIClient
from datasets import Dataset
import json
from dotenv import load_dotenv
import os
from typing import Dict, List
from sklearn.metrics import classification_report
import random  # 引入 random 来确保测试集划分的随机性

# --- 1. 配置和 LLM 客户端 ---
load_dotenv()

# 请确保您的 .env 文件配置正确
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 初始化 LLM 客户端
llm = OpenAIClient(
    model="gpt-5",
    temperature=0.1,
    top_p=0.9,
    max_tokens=500  # 减少 token 数量，只需要简短的 JSON 响应
)

# --- 2. 专门用于 DSCD 的系统提示 ---
CONFLICT_SYSTEM_PROMPT = (
    "You are an expert in detecting contradictions within a single document.\n"
    "Your task is to analyze the entire document and determine if it contains any self-contradictory information, "
    "where two or more statements are logically incompatible with each other.\n"
    "Respond with a single JSON object with two keys:\n"
    '{"is_contradictory": <boolean>, "reason": "<brief explanation of the conflict, or why none was found>"}\n'
    "Respond with only the JSON object—no explanations, no markdown, no extra text."
)


# --- 3. 数据加载函数 (展平 ContraDOC 结构) ---
def load_contradoc_data(file_path: str) -> List[Dict]:
    """手动加载 ContraDOC JSON 文件并展平结构"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: ContraDOC file not found at {file_path}")
        return []

    flattened_data = []

    # 处理 "pos" (有矛盾, label=1)
    if "pos" in data:
        for unique_id, doc_data in data["pos"].items():
            flattened_data.append({
                "document": doc_data["text"],
                "label": 1,
                "unique_id": unique_id
            })

    # 处理 "neg" (无矛盾, label=0)
    if "neg" in data:
        for unique_id, doc_data in data["neg"].items():
            flattened_data.append({
                "document": doc_data["text"],
                "label": 0,
                "unique_id": unique_id
            })

    return flattened_data


# --- 4. GPT API 调用和解析函数 ---
def detect_conflict_with_gpt(llm_client: OpenAIClient, document_text: str, system_prompt: str) -> int:
    """使用 GPT 模型检测文档内冲突，并返回整数标签 (0 或 1)"""
    try:
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt)

        # 清理并解析 JSON 响应
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()

        response_json = json.loads(response_text)

        # 返回预测的标签 (1 for True/Contradictory, 0 for False/Non-Contradictory)
        # 使用 get 方法确保键不存在时也能有默认值 False
        return int(response_json.get("is_contradictory", False))

    except Exception as e:
        # 在 API 调用失败或 JSON 解析失败时，输出错误并返回默认值
        print(f"Error processing document (returning 0): {e}. Raw Response: {response_text[:100]}...")
        return 0


# --- 5. 主执行逻辑 ---
if __name__ == '__main__':
    # 限制 LLM 调用的样本数，因为 LLM 调用较慢且收费
    LIMIT = 20

    data_files = {
        "train": "../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"
    }

    print("🌍 正在加载 ContraDOC 数据集并展平结构...")
    flattened_data = load_contradoc_data(data_files["train"])

    if not flattened_data:
        print("无法加载数据，程序退出。")
        exit()

    raw_dataset = Dataset.from_list(flattened_data)

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
        print(f"Processing {i + 1}/{len(test_dataset)}: ID {example['unique_id']}...")

        prediction = detect_conflict_with_gpt(llm, example['document'], CONFLICT_SYSTEM_PROMPT)
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