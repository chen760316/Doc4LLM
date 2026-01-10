"""
使用的数据集：ContraDOC
使用的基准："gpt-4.1-mini"

每15篇文档在如下四个问题的评估上一起用一次gpt，效率高

### RQ1：在LLM的某些领域，自我矛盾更难被发现吗？
我们从三个领域构建了一个文档语料库CONTRADOC，以测试其在各种上下文中的性能。我们使用CNN-DailyMail数据集(赫尔曼等人，2015年)处理新闻文章，使用NarativeQA(Koˇcisk`y等人，2018年)处理故事，使用Wikitext(Merity等人，2016年)处理维基百科文档(详细信息见附录A)。对于每个文档，都插入了一个自相矛盾的内容。

### RQ2：在LLM的较长文档中是否更难发现自相矛盾？
CONTRADOC中的文档范围从100个令牌到2200个令牌，帮助我们学习更长和更短的文档。表6显示了我们的数据集关于文档长度的详细细分(以令牌为单位)。

### RQ3：自相矛盾会不会出现在更远的地方，对LLM来说更难发现？
为了测试上下文长度对模型性能的影响，我们引入了相互之间存在不同距离的矛盾。我们将出现范围定义为：在一个句子内出现矛盾的实例被标记为内部，而矛盾语句出现在四句话以下的实例被标记为局部，最后，矛盾出现在四句话以上的实例被标记为全局。我们的数据集包含73、220和155个具有内部、局部和全局矛盾的文档。

### RQ4：对于LLM来说，某些类型的自我矛盾是否比其他类型的更难发现？
De Marneffe等人。(2008)将矛盾定义为两大类，内容矛盾和词汇矛盾，Wu et al.(2022)类似于句子层面的矛盾检测，定义了六种类型的自我矛盾。我们将重点放在内容上并对其进行扩展，以构建更细粒度的分类法。引入更完整的类型选择来研究：CONTRADOC中的每个文档都标记了以下八种类型的自我矛盾中的一种或多种：否定、数字、内容、观点、情感、事实、关系和因果。
"""
import json
import random
import re
import ast
import torch
from evaluate import load
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from knowledge_graph_maker import OpenAIClient  # 假设已安装
from datasets import Dataset  # 假设已安装
from dotenv import load_dotenv
import os
import collections

# --- 1. 配置和 LLM 客户端 ---
load_dotenv()


# 请确保您的 .env 文件配置正确 (此处仅为示例，实际运行请确保环境变量已设置)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- 关键修改点：将 MockLLMClient 定义移到顶层 ---
class MockLLMClient:
    def generate(self, user_message: str, system_message: str):
        # 模拟模型随机判断（实际模型应该更智能）
        if random.random() < 0.5:
            return '{"is_contradictory": true}'
        else:
            return '{"is_contradictory": false}'


# 初始化 LLM 客户端
try:
    llm = OpenAIClient(
        model="gpt-5.2",  # 假设这是一个可用且便宜的模型
        temperature=0.1,
        top_p=0.9,
        max_tokens=100  # 只需要 JSON 输出，可以更少
    )
except Exception:  # 捕获 Exception 确保能处理各种错误
    llm = MockLLMClient()
    print("WARNING: Using MockLLMClient due to potential environment issues.")


# --- 2. 核心 LLM 冲突检测函数 (未改动，现在可以找到 MockLLMClient) ---


# 定义批量大小
BATCH_SIZE = 15


# --- 新增：批量冲突检测函数 ---
def detect_conflict_with_gpt_batch(llm_client: Any, batch_data: List[Dict[str, Any]], system_prompt: str) -> List[int]:
    """
    使用 GPT 模型批量检测文档内冲突，并返回整数标签列表。
    改进：增强了步骤 4 的 JSON 解析鲁棒性。
    """
    # ----------------------------------------------------------------------
    # 步骤 1: 构建批量输入的用户消息 (User Message)
    # ----------------------------------------------------------------------
    batch_input_list = []
    for i, doc_info in enumerate(batch_data):
        # 将每个文档编号并格式化
        batch_input_list.append(f"DOCUMENT_{i + 1}:\n{doc_info['text']}\n---END_DOCUMENT_{i + 1}---")

    # 组合成最终的用户消息
    document_text = "\n\n".join(batch_input_list)

    # ----------------------------------------------------------------------
    # 步骤 2: 构建批量系统提示 (System Prompt)
    # ----------------------------------------------------------------------

    # 关键：要求模型返回一个 JSON 列表，列表长度必须与输入的文档数一致
    system_prompt_json = f"""
    {system_prompt}.
    You have been provided with {len(batch_data)} documents, each clearly marked (e.g., DOCUMENT_1, DOCUMENT_2).
    Your output MUST be a single JSON object containing a list under the key 'contradictions'. 
    This list must contain {len(batch_data)} objects, one for each input document, in the exact input order. 
    Each object must have two keys: 'document_id' (e.g., 'DOCUMENT_1') and 'is_contradictory' (true/false).

    Example Output Structure for 2 documents:
    {{
        "contradictions": [
            {{"document_id": "DOCUMENT_1", "is_contradictory": true}},
            {{"document_id": "DOCUMENT_2", "is_contradictory": false}}
        ]
    }}
    Do not include any other text or explanations.
    """

    # ----------------------------------------------------------------------
    # 步骤 3: 调用 LLM
    # ----------------------------------------------------------------------

    # 假设 MockLLMClient 和 llm_client 定义在外部
    if isinstance(llm_client, MockLLMClient):
        # Mock 客户端的批量处理：为每个文档生成一个随机结果
        mock_results = []
        for i in range(len(batch_data)):
            is_contradictory = random.random() < 0.5
            mock_results.append({"document_id": f"DOCUMENT_{i + 1}", "is_contradictory": is_contradictory})
        response_text = json.dumps({"contradictions": mock_results})
    else:
        # 实际 LLM 调用
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt_json)

    # ----------------------------------------------------------------------
    # 步骤 4: 改进的解析响应 (增强鲁棒性)
    # ----------------------------------------------------------------------
    parsed_predictions = []
    try:
        response_text = response_text.strip()

        # 1. 清理：移除代码块标记 (```json...)
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()

        # 2. 查找：使用正则表达式找到最外层的 JSON 对象
        # re.DOTALL 确保 . 匹配换行符，匹配从第一个 { 到最后一个 }
        match = re.search(r'(\{.*?\})', cleaned_text, re.DOTALL)

        if not match:
            # 如果找不到 JSON，或者模型只返回了部分文本
            print(f"WARNING: No complete JSON found in response. Returning all 0s.")
            print(f"Raw Response Head: {response_text[:300]}...")  # 打印更多原始响应头
            return [0] * len(batch_data)

        # 3. 解析
        json_string = match.group(0).lower()  # 提取匹配到的 JSON 字符串并转小写
        response_json = json.loads(json_string)

        # 4. 提取结果
        results_list = response_json.get("contradictions", [])

        # 验证结果数量是否匹配
        if len(results_list) != len(batch_data):
            print(f"WARNING: Batch size mismatch. Expected {len(batch_data)}, got {len(results_list)}. Filling with 0s.")
            return [0] * len(batch_data)

        # 提取 is_contradictory 字段并转换为整数 1/0
        for item in results_list:
            # 兼容布尔值、字符串（如 "true"）和任何非预期值
            is_contradictory = item.get("is_contradictory", False)
            if isinstance(is_contradictory, str):
                parsed_predictions.append(1 if is_contradictory.strip() == "true" else 0)
            else:
                parsed_predictions.append(int(is_contradictory))

        return parsed_predictions

    except Exception as e:
        # 捕获所有解析错误，并打印更多信息
        print(f"CRITICAL PARSE ERROR (returning all 0s): {e}.")
        print(f"Failed to parse text (Head): {response_text[:300]}...")
        return [0] * len(batch_data)


# --- 3. 数据加载与处理 (未改动) ---

def load_docs_evis_infos(path: str, pos_only: bool = False) -> List[Dict[str, Any]]:
    """
    加载数据，并包含 RQ 所需的元数据。
    注意: 实际运行需确保 ContraDoc.json 包含 'domain', 'length_group', 'distance_group', 'type' 等元数据。
    """
    # 模拟 ContraDoc.json 数据结构 (实际应从文件中加载)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found at {path}. Using simulated data structure for demonstration.")
        # 模拟数据：关键在于添加了用于分组的元数据
        data = {
            "pos": {
                "p1": {"text": "Doc 1. The sun is red.", "evidence": "The sun is red.", "domain": "Wikitext", "length_group": "Short", "distance_group": "Internal", "type": "Fact"},
                "p2": {"text": "Doc 2. I like blue. But red is better.", "evidence": "I like blue.", "domain": "NarrativeQA", "length_group": "Medium", "distance_group": "Local", "type": "Opinion"},
                "p3": {"text": "Doc 3. The year is 2025. It's 2024.", "evidence": "The year is 2025.", "domain": "CNN-DailyMail", "length_group": "Long", "distance_group": "Global", "type": "Number"},
                "p4": {"text": "Doc 4. Another doc.", "evidence": "Another doc.", "domain": "Wikitext", "length_group": "Short", "distance_group": "Internal", "type": "Fact"},
            },
            "neg": {
                "n1": {"text": "Doc 5. No conflict here.", "evidence": None, "domain": "Wikitext", "length_group": "Short", "distance_group": "Internal", "type": "None"},
                "n2": {"text": "Doc 6. Safe content.", "evidence": None, "domain": "CNN-DailyMail", "length_group": "Medium", "distance_group": "Global", "type": "None"},
            }
        }

    all_info = []

    # Process Positive Examples
    for key, value in data["pos"].items():
        value["label"] = 1  # True for contradiction
        value["id"] = key
        # 确保关键分组字段存在，否则设为 Unknown
        value.setdefault("domain", "Unknown")
        value.setdefault("length_group", "Unknown")
        value.setdefault("distance_group", "Unknown")
        value.setdefault("type", "Unknown")
        all_info.append(value)

    if pos_only:
        return all_info

    # Process Negative Examples
    for key, value in data["neg"].items():
        value["label"] = 0  # False for non-contradiction
        value["id"] = key
        value["evidence"] = None
        # 确保关键分组字段存在
        value.setdefault("domain", "Unknown")
        value.setdefault("length_group", "Unknown")
        value.setdefault("distance_group", "Unknown")
        value.setdefault("type", "Unknown")
        all_info.append(value)

    random.shuffle(all_info)  # 随机打乱，避免模型学习顺序
    return all_info


# --- 4. 评估逻辑：按分组计算指标 (未改动) ---

def get_llm_responses(llm_client: Any, data: List[Dict[str, Any]], system_prompt: str) -> List[int]:
    """
    修改后的函数：使用批量处理获取所有文档的 LLM 预测结果。
    """
    all_predictions = []
    total_docs = len(data)

    # 将数据分成大小为 BATCH_SIZE 的批次
    # 使用 NumPy 或简单的循环来切分数据
    num_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n--- Running LLM Inference on {total_docs} documents in {num_batches} batches (Batch Size: {BATCH_SIZE}) ---")

    for i in range(num_batches):
        start_index = i * BATCH_SIZE
        end_index = min((i + 1) * BATCH_SIZE, total_docs)

        batch_data = data[start_index:end_index]

        print(f"Processing batch {i + 1}/{num_batches} (Docs {start_index + 1}-{end_index})...")

        # 调用新的批量检测函数
        batch_predictions = detect_conflict_with_gpt_batch(llm_client, batch_data, system_prompt)

        all_predictions.extend(batch_predictions)

    return all_predictions

def calculate_metrics_by_group(data: List[Dict[str, Any]], predictions: List[int], group_key: str):
    """按指定的元数据键对数据进行分组，并对每个组计算分类指标"""

    grouped_data = collections.defaultdict(lambda: {"labels": [], "predictions": []})

    for doc_info, pred in zip(data, predictions):
        group_name = doc_info[group_key]
        grouped_data[group_name]["labels"].append(doc_info["label"])
        grouped_data[group_name]["predictions"].append(pred)

    print(f"\n--- Results Grouped by: {group_key} (Answering RQ) ---")
    for group_name, group_data in grouped_data.items():
        labels = group_data["labels"]
        preds = group_data["predictions"]

        # 避免除以零错误
        if not labels: continue

        # 计算指标
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        a = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)

        print(f"\n>> Group: {group_name} (N={len(labels)})")
        print(f"   Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}, Accuracy: {a:.4f}")


# --- 5. 主执行逻辑 (未改动) ---

if __name__ == "__main__":
    # 基础系统提示，要求模型仅判断是否矛盾 (用于 Yes/No 任务)
    system_prompt = "You are an expert contradiction detection system. Your task is to determine whether the provided document contains any self-contradictions."

    # 假设 ContraDoc.json 位于项目根目录的 datasets/ContraDOC/ 下
    data_path = "../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"

    # 1. 加载所有数据和元数据
    # 注意：模拟数据只有 6 条，因此只会运行 1 个批次
    all_data_info = load_docs_evis_infos(data_path, pos_only=False)

    # 2. 获取 LLM 的预测结果 (现在是批量处理)
    all_predictions = get_llm_responses(llm, all_data_info, system_prompt)

    # 3. 总体性能评估 (作为基线)
    overall_labels = [x["label"] for x in all_data_info]
    print("\n--- Overall Performance (Baseline) ---")
    print(classification_report(overall_labels, all_predictions, target_names=["No Contradiction (0)", "Contradiction (1)"], zero_division=0))

    # 4. 按 RQ 分组进行细分评估
    calculate_metrics_by_group(all_data_info, all_predictions, group_key="domain")
    calculate_metrics_by_group(all_data_info, all_predictions, group_key="length_group")
    calculate_metrics_by_group(all_data_info, all_predictions, group_key="distance_group")
    calculate_metrics_by_group(all_data_info, all_predictions, group_key="type")