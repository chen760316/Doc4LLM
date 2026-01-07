"""
使用的数据集：ContraDOC
使用的基准："gpt-4.1-mini"

每篇文档在如下四个问题的评估上都需要调用一次gpt，效率太低：

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
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

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
        model="gpt-4.1-mini",  # 假设这是一个可用且便宜的模型
        temperature=0.1,
        top_p=0.9,
        max_tokens=100  # 只需要 JSON 输出，可以更少
    )
except Exception:  # 捕获 Exception 确保能处理各种错误
    llm = MockLLMClient()
    print("WARNING: Using MockLLMClient due to potential environment issues.")


# --- 2. 核心 LLM 冲突检测函数 (未改动，现在可以找到 MockLLMClient) ---

def detect_conflict_with_gpt(llm_client: Any, document_text: str, system_prompt: str) -> int:
    """使用 GPT 模型检测文档内冲突，并返回整数标签 (1 for YES, 0 for NO)"""
    # 针对 Yes/No 任务，简化 system_prompt
    system_prompt_json = f"{system_prompt}. Your output MUST be a single JSON object: {{ \"is_contradictory\": true/false }}. Do not include any explanations."

    # 避免在 MockLLMClient 上调用 generate (因为它是 Mock)
    # 修复后的代码，isinstance(llm_client, MockLLMClient) 不会报 NameError
    if isinstance(llm_client, MockLLMClient):
        # 注意：MockLLMClient.generate 的签名是 (self, user_message, system_message)
        # 这里需要调整调用方式，或者调整 MockLLMClient 的 generate 方法。
        # 按照您提供的 MockLLMClient.generate 签名，它接收两个位置参数：
        response_text = llm_client.generate(document_text, system_prompt_json)
    else:
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt_json)

    try:
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text.lower())  # 转换为小写以便安全解析

        # 匹配 true/false 字符串，如果键不存在则默认为 False
        is_contradictory = response_json.get("is_contradictory", False)
        if isinstance(is_contradictory, str):
            return 1 if is_contradictory == "true" else 0
        return int(is_contradictory)

    except Exception as e:
        # 在 API 调用失败或 JSON 解析失败时，输出错误并返回默认值 0 (Non-Contradictory)
        # print(f"Error processing document (returning 0): {e}. Raw Response: {response_text[:100]}...")
        return 0


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
    """获取所有文档的 LLM 预测结果"""
    predictions = []
    print(f"\n--- Running LLM Inference on {len(data)} documents ---")
    for i, doc_info in enumerate(data):
        if i % 10 == 0:
            print(f"Processing document {i}/{len(data)}...")

        # 负例：实际测试中，需要对所有样本运行 LLM
        prediction = detect_conflict_with_gpt(llm_client, doc_info["text"], system_prompt)

        predictions.append(prediction)
    return predictions


def calculate_metrics_by_group(data: List[Dict[str, Any]], predictions: List[int], group_key: str) -> Dict[str, Dict[str, float]]:
    """按指定的元数据键对数据进行分组，计算指标，并返回分组结果"""

    grouped_data = collections.defaultdict(lambda: {"labels": [], "predictions": []})
    group_results = {} # 新增：用于存储返回结果的字典

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

        # 存储结果到字典
        group_results[group_name] = {
            "N": len(labels),
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "Accuracy": a
        }

        # 保持打印输出以便实时监控
        print(f"\n>> Group: {group_name} (N={len(labels)})")
        print(f"   Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}, Accuracy: {a:.4f}")

    return group_results # 返回结构化的结果


def save_full_report(results: Dict[str, Any], model_name: str, filename_prefix: str = "full_report"):
    """将完整的评估结果字典保存到 JSON 文件"""
    filename = f"{filename_prefix}_{model_name}.json"
    try:
        # 创建输出目录 (如果需要)
        # os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n\n✅ 完整的评估报告已成功保存到: {filename}")
    except Exception as e:
        print(f"\n❌ 错误: 无法写入报告文件 {filename}. 详细信息: {e}")


# --- 5. 主执行逻辑 (未改动) ---

if __name__ == "__main__":
    # 配置
    MODEL_NAME = "GPT_Single_Call"  # 为本次运行定义一个模型名称
    system_prompt = "You are an expert contradiction detection system. Your task is to determine whether the provided article contains any self-contradictions. Respond only with a single JSON object."
    data_path = "../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"

    # 1. 加载所有数据和元数据
    all_data_info = load_docs_evis_infos(data_path, pos_only=False)

    # 2. 获取 LLM 的预测结果
    all_predictions = get_llm_responses(llm, all_data_info, system_prompt)

    # 3. 总体性能评估 (作为基线)
    overall_labels = [x["label"] for x in all_data_info]

    # 使用 classification_report 的 output_dict 模式捕获详细指标
    overall_report = classification_report(overall_labels, all_predictions,
                                           target_names=["No Contradiction (0)", "Contradiction (1)"],
                                           zero_division=0, output_dict=True)

    print("\n--- Overall Performance (Baseline) ---")
    # 打印可读的报告
    print(classification_report(overall_labels, all_predictions, target_names=["No Contradiction (0)", "Contradiction (1)"], zero_division=0))

    # 4. 按 RQ 分组进行细分评估，并捕获结果字典

    full_report_data = {
        "model_name": MODEL_NAME,
        "total_samples": len(all_data_info),
        "Overall_Metrics": overall_report,

        # 捕获分组结果
        "RQ1_Domain": calculate_metrics_by_group(all_data_info, all_predictions, group_key="domain"),
        "RQ2_Length": calculate_metrics_by_group(all_data_info, all_predictions, group_key="length_group"),
        "RQ3_Distance": calculate_metrics_by_group(all_data_info, all_predictions, group_key="distance_group"),
        "RQ4_Type": calculate_metrics_by_group(all_data_info, all_predictions, group_key="type"),
    }

    # 5. 调用新增的保存函数
    save_full_report(full_report_data, MODEL_NAME)