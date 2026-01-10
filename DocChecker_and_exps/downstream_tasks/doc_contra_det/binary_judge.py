"""
使用的数据集：ContraDoc
使用的基准："gpt-4.1-mini"
评估模型最直接的方法是测试它们区分正反两个例子的能力。
我们通过简单地要求模型提供关于文档d是否自相矛盾的判断来做到这一点。
在这种情况下，我们在CONTRADOC上对模型进行评估。我们使用标准的精确度、召回率、F1分数和准确性指标来评估模型的二元判断，记为j(D)。
"""
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
import difflib
import random
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import collections # 用于计数和分组

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


# --- 2. 核心检测函数 ---

def detect_conflict_with_gpt(llm_client: Any, document_text: str, system_prompt: str) -> Tuple[int, str]:
    """
    使用 GPT 模型检测文档内冲突，返回 (标签, 解释)
    标签: 1 for YES (矛盾), 0 for NO (非矛盾)
    解释: 模型的文字解释
    """

    # 针对 Yes/No 任务，强制要求 JSON 输出，包含解释字段
    system_prompt_json = (
        f"{system_prompt}. "
        "Your output MUST be a single JSON object: { \"is_contradictory\": true/false, \"explanation\": \"[Your reason or evidence for the decision.]\" }. "
        "Do not include any other text outside the JSON block. Output only the JSON."
    )

    default_explanation = "无法获取模型响应或 JSON 解析失败。"

    try:
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt_json)

    except TypeError:
        # 如果是 MockLLMClient (假设)，需要修改其签名或在此处处理
        return 0, default_explanation
    except Exception:
        # API调用失败等情况
        return 0, default_explanation

    try:
        # 清理并解析 JSON
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text.lower())

        # 尝试获取标签
        is_contradictory = response_json.get("is_contradictory", False)
        if isinstance(is_contradictory, str):
            label = 1 if is_contradictory == "true" else 0
        else:
            label = int(is_contradictory)

        # 尝试获取解释，并清理可能的引号
        explanation = response_json.get("explanation", default_explanation).strip().strip('"').strip("'")

        return label, explanation

    except Exception:
        # JSON 解析失败，返回默认值
        return 0, default_explanation


# --- 3. 模拟/加载数据集函数 ---

def load_and_sample_contradoc_data(full_data: Dict, num_pos: int = 30, num_neg: int = 30) -> List[Dict[str, Any]]:
    """
    从 ContraDoc JSON 结构中加载数据，并按照要求进行抽样。
    - 正例 (pos): 尽量覆盖不同的 'contra_type'。
    - 负例 (neg): 随机抽取。
    """
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    # 1. 处理正例 (Positives)
    pos_list = []

    # 按照 contra_type 分组
    pos_grouped = collections.defaultdict(list)
    for unique_id, doc_info in pos_data.items():
        # 将文档信息添加到所有它拥有的 contra_type 组中
        types = doc_info.get("contra_type", ["Unknown"])
        for c_type in types:
            pos_grouped[c_type].append({
                "unique_id": unique_id,
                "text": doc_info["text"],
                "label": 1,
                "type": c_type  # 用于抽样辅助
            })

    # 尝试均匀抽样

    # 获取所有类型和每种类型拥有的文档数
    all_types = list(pos_grouped.keys())
    type_counts = {t: len(pos_grouped[t]) for t in all_types}

    # 计算每种类型应该抽取的“基础”数量
    base_sample_per_type = num_pos // len(all_types)
    remaining_to_sample = num_pos
    sampled_ids = set()  # 确保不重复抽取同一个文档

    # 优先抽取基础数量
    for c_type in all_types:
        # 实际抽取数量不能超过该类型文档总数，也不能超过基础配额
        num_to_sample = min(base_sample_per_type, len(pos_grouped[c_type]))

        # 随机选择未被抽取的文档
        available_docs = [d for d in pos_grouped[c_type] if d['unique_id'] not in sampled_ids]

        if available_docs:
            sample = random.sample(available_docs, min(num_to_sample, len(available_docs)))
            for doc in sample:
                if doc['unique_id'] not in sampled_ids:
                    pos_list.append(doc)
                    sampled_ids.add(doc['unique_id'])
                    remaining_to_sample -= 1

    # 填补剩余的配额 (如果还有剩余名额未达到 num_pos)
    # 从所有未被抽样的文档中随机抽取
    if remaining_to_sample > 0:
        all_remaining_docs = []
        for c_type in all_types:
            available_docs = [d for d in pos_grouped[c_type] if d['unique_id'] not in sampled_ids]
            all_remaining_docs.extend(available_docs)

        # 去重（因为一个文档可能属于多种类型）
        unique_remaining_docs = list({d['unique_id']: d for d in all_remaining_docs}.values())

        if unique_remaining_docs:
            final_sample = random.sample(unique_remaining_docs, min(remaining_to_sample, len(unique_remaining_docs)))
            pos_list.extend(final_sample)

    # 确保最终抽样数量不超过 num_pos
    pos_list = pos_list[:num_pos]

    # 2. 处理负例 (Negatives)
    neg_available = []
    for unique_id, doc_info in neg_data.items():
        neg_available.append({
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 0  # 非矛盾
        })

    # 随机抽样
    num_to_sample_neg = min(num_neg, len(neg_available))
    neg_list = random.sample(neg_available, num_to_sample_neg)

    print(f"\n--- 数据抽样统计 ---")
    print(f"正例 (矛盾) 抽取数: {len(pos_list)} / 目标 {num_pos}")
    print(f"负例 (非矛盾) 抽取数: {len(neg_list)} / 目标 {num_neg}")

    # 3. 合并和返回
    dataset = pos_list + neg_list
    random.shuffle(dataset)
    return dataset


# --- 4. 结果保存辅助函数 ---

def save_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存评估指标和详细的文档结果到本地文件。"""
    base_filename = f"ContraDoc_Binary_Eval_{model_name}"

    # 1. 保存详细结果 (JSON)
    json_path = f"{base_filename}_details.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"\n✨ 详细评估结果已保存到: {json_path}")

    # 2. 保存指标结果 (TXT)
    txt_path = f"{base_filename}_metrics.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ContraDoc 二元检测评估指标 ({model_name}) ---\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 40 + "\n")
        f.write("详细分类报告:\n")
        f.write(metrics.get("classification_report", "N/A"))

    print(f"✨ 最终评估指标已保存到: {txt_path}")


# --- 5. 评估主函数 (已修改以收集结果) ---

def evaluate_binary_detection(dataset: List[Dict[str, Any]], llm_client: Any, system_prompt: str):
    """
    对数据集执行二元检测并计算评估指标，并将结果保存在本地。
    """
    true_labels = []
    pred_labels = []
    detailed_results = []  # 用于保存每个文档的详细信息

    model_name = getattr(llm_client, 'model', 'MockLLM')  # 获取模型名称
    print(f"\n--- 开始评估 ({len(dataset)} 个文档, 模型: {model_name}) ---")

    for i, item in enumerate(dataset):
        doc_text = item["text"]
        unique_id = item["unique_id"]
        true_label = item["label"]

        # 获取模型的预测结果和解释
        predicted_label, explanation = detect_conflict_with_gpt(llm_client, doc_text, system_prompt)

        true_labels.append(true_label)
        pred_labels.append(predicted_label)

        # 收集详细结果
        result_entry = {
            "unique_id": unique_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "explanation": explanation,
            "is_correct": (true_label == predicted_label)
        }
        detailed_results.append(result_entry)

        # 打印自相矛盾的文档及其解释
        if predicted_label == 1:
            print(f"\n*** ❗ 检测到矛盾 (ID: {unique_id}) ***")
            print(f"真实标签: {'矛盾 (1)' if true_label == 1 else '非矛盾 (0)'}")
            print(f"模型解释: {explanation}")

        if (i + 1) % 10 == 0 or i == len(dataset) - 1:
            print(f"✅ 已处理 {i + 1}/{len(dataset)} 个文档...")

    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    # 生成分类报告文本
    cls_report = classification_report(true_labels, pred_labels, target_names=["Non-Contradictory (0)", "Contradictory (1)"], zero_division=0)

    print("\n--- 评估结果 (Metrics) ---")
    print(f"Accuracy (准确性): {accuracy:.4f}")
    print(f"Precision (精确度): {precision:.4f}")
    print(f"Recall (召回率): {recall:.4f}")
    print(f"F1 Score (F1 分数): {f1:.4f}")
    print("\n--- 详细分类报告 ---")
    print(cls_report)

    # 准备保存的指标字典
    metrics_to_save = {
        "Model": model_name,
        "Total_Samples": len(dataset),
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1_Score": f"{f1:.4f}",
        "classification_report": cls_report
    }

    # 保存结果到文件
    save_results_to_file(metrics_to_save, detailed_results, model_name)


# --- 6. 执行脚本 ---

if __name__ == "__main__":

    try:
        # 假设文件路径正确
        with open("../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json", 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print("Error: ContraDoc.json not found. Please check the file path.")
        exit()
    except json.JSONDecodeError:
        print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
        exit()

    # 使用新的抽样函数加载数据集
    contradoc_dataset = load_and_sample_contradoc_data(full_data, num_pos=30, num_neg=30)

    # 定义系统提示
    SYSTEM_PROMPT = (
        "You are an expert natural language processing model specialized in identifying self-contradictory "
        "statements within a single document. Carefully read the provided document. "
        "Determine if the document contains any internal contradictions, meaning two or more sentences "
        "in the text present facts or information that cannot simultaneously be true. "
        "If it is self-contradictory, output true. Otherwise, output false. "
        "The 'explanation' field must state the contradictory evidence or the reason for the non-contradictory decision."
    )

    # 执行评估
    evaluate_binary_detection(contradoc_dataset, llm, SYSTEM_PROMPT)