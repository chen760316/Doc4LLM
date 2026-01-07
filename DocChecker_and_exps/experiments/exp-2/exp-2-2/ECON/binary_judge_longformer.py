# -*- coding: utf-8 -*-
import json
import os
import random
from typing import Dict, List, Any, Tuple
import collections
import time
import sys

# --- 引入 Hugging Face 库 ---
# 必须安装: pip install transformers torch sklearn
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

    # 假设使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("❌ 错误: 缺少必要的库 (transformers, torch, 或 sklearn)。请运行 pip install transformers torch sklearn")
    sys.exit(1)

DEFAULT_HF_MODEL = "roberta-large-mnli"
print(f"!!! 警告: 尝试加载微调 NLI 模型: {DEFAULT_HF_MODEL}。长文本 (>512) 将被截断。")
print(f"!!! 请确保您的评估逻辑 (predict_conflict_with_longformer) 能够正确处理 3 个标签的输出，并映射到 0/1。")

print("--- 1. 正在加载 Hugging Face NLI 模型 ---")
try:
    # 自动加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_HF_MODEL)

    # 强制将模型移到 GPU
    model.to(device)
    model.eval()
    print(f"✅ NLI 模型 ({DEFAULT_HF_MODEL}) 加载成功，设备: {device}")

    # 获取模型的标签数量和ID到标签名的映射
    num_labels = model.config.num_labels
    id2label = model.config.id2label
    print(f"✅ 模型标签数量: {num_labels}，标签映射: {id2label}")

except Exception as e:
    print(f"❌ 错误: 模型加载失败: {e}")
    model = None
    num_labels = 2
    id2label = {0: 'LABEL_0', 1: 'LABEL_1'}


# --- 2. 核心检测函数 (NLI 3标签 -> 2标签映射) ---

def predict_conflict_with_longformer(document_text: str) -> Tuple[int, str]:
    """
    使用 NLI 模型进行分类推理。
    将 NLI 3 个标签中的 "Contradiction" 映射为 1 (矛盾)，其他映射为 0 (非矛盾)。
    """
    if model is None:
        return 0, "模型未加载或初始化失败。"

    try:
        # 1. 编码输入 (RoBERTa 默认为 512 max_length, 强制截断长文本)
        inputs = tokenizer(
            document_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # *** 关键修改：使用 RoBERTa 的最大长度 ***
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()

        # --- 标签映射 ---

        # 1. 尝试从模型配置中获取标签名
        # RoBERTa-large-mnli 默认的 id2label 是: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        predicted_label_name = id2label.get(predicted_class_id, 'UNKNOWN')

        # 2. 映射到二元输出 (目标标签 1=Contradictory, 0=Non-Contradictory)
        is_contradictory = 'contradiction' in predicted_label_name.lower()

        if is_contradictory:
            label = 1  # NLI 模型预测为 Contradiction -> 二元 1
        else:
            label = 0  # NLI 模型预测为 Entailment 或 Neutral -> 二元 0

        # -------------------------------

        explanation = f"NLI 模型分类结果: {predicted_label_name} (ID: {predicted_class_id}). 二元映射为 {'矛盾 (Contradictory)' if label == 1 else '非矛盾 (Non-Contradictory)'}."

        return label, explanation

    except Exception as e:
        return 0, f"NLI 推理过程中发生异常: {e}"


# --- 4. 结果保存辅助函数 (保持不变) ---

def save_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存评估指标和详细的文档结果到本地文件。"""

    SAVE_DIR = "report/binary_judge_report_longformer"
    os.makedirs(SAVE_DIR, exist_ok=True)

    base_filename = f"ContraDoc_Binary_Eval_{model_name.replace('/', '_')}"

    # 1. 保存详细结果 (JSON)
    json_path = os.path.join(SAVE_DIR, f"{base_filename}_details.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"\n✨ 详细评估结果已保存到: {json_path}")

    # 2. 保存指标结果 (TXT)
    txt_path = os.path.join(SAVE_DIR, f"{base_filename}_metrics.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ContraDoc 二元检测评估指标 ({model_name}) ---\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy (准确性): {metrics.get('Accuracy', 'N/A')}\n")
        f.write(f"Precision (精确度): {metrics.get('Precision', 'N/A')}\n")
        f.write(f"Recall (召回率): {metrics.get('Recall', 'N/A')}\n")
        f.write(f"F1 Score (F1 分数): {metrics.get('F1_Score', 'N/A')}\n")
        f.write("-" * 40 + "\n")
        f.write("详细分类报告:\n")
        f.write(metrics.get("classification_report", "N/A"))

    print(f"✨ 最终评估指标已保存到: {txt_path}")


# --- 5. 评估主函数 (单线程 - 保持不变) ---

def evaluate_binary_detection_single_threaded(dataset: List[Dict[str, Any]]):
    """
    对数据集执行二元检测并计算评估指标，使用单线程 (适合本地模型推理)。
    """
    true_labels = []
    pred_labels = []
    detailed_results = []

    model_name = DEFAULT_HF_MODEL
    total_samples = len(dataset)
    print(f"\n--- 开始评估 ({total_samples} 个文档, 模型: {model_name}, 使用单线程/本地推理) ---")

    start_time = time.time()

    for processed_count, item in enumerate(dataset, 1):
        doc_text = item["text"]
        unique_id = item["unique_id"]
        true_label = item["label"]

        predicted_label, explanation = predict_conflict_with_longformer(doc_text)

        result_entry = {
            "unique_id": unique_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "explanation": explanation,
            "is_correct": (true_label == predicted_label)
        }

        true_labels.append(true_label)
        pred_labels.append(predicted_label)
        detailed_results.append(result_entry)

        if predicted_label == 1:
            print(f"\n*** ❗ 检测到矛盾 (ID: {unique_id}) - 进度: {processed_count}/{total_samples} ***")
            print(f"真实标签: {'矛盾 (1)' if true_label == 1 else '非矛盾 (0)'}")
            print(f"模型解释: {explanation}")
        else:
            if processed_count % 10 == 0 or processed_count == total_samples:
                print(f"✅ 已处理 {processed_count}/{total_samples} 个文档... ({time.time() - start_time:.2f}s)")

    end_time = time.time()
    print(f"\n评估完成！总耗时: {end_time - start_time:.2f} 秒.")

    # --- 结果汇总和指标计算 ---
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    cls_report = classification_report(true_labels, pred_labels, target_names=["Non-Contradictory (0)", "Contradictory (1)"], zero_division=0)

    print("\n--- 评估结果 (Metrics) ---")
    print(f"Accuracy (准确性): {accuracy:.4f}")
    print(f"Precision (精确度): {precision:.4f}")
    print(f"Recall (召回率): {recall:.4f}")
    print(f"F1 Score (F1 分数): {f1:.4f}")
    print("\n--- 详细分类报告 ---")
    print(cls_report)

    metrics_to_save = {
        "Model": model_name,
        "Total_Samples": len(dataset),
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1_Score": f"{f1:.4f}",
        "classification_report": cls_report
    }

    save_results_to_file(metrics_to_save, detailed_results, model_name)


def load_all_contradoc_data(full_data: Dict) -> List[Dict[str, Any]]:
    """
    从 ContraDoc JSON 结构中加载所有正例 (pos) 和负例 (neg) 数据，不进行采样。

    Args:
        full_data (Dict): 加载的 ContraDoc 完整 JSON 数据结构。
                          预期格式: {"pos": {...}, "neg": {...}}

    Returns:
        List[Dict[str, Any]]: 包含所有文档的列表，每个字典包含 unique_id, text, 和 label (1 或 0)。
    """
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    all_data_list = []

    # 1. 加载所有正例 (Positives)
    # 标签 1: 矛盾 (Contradictory)
    for unique_id, doc_info in pos_data.items():
        item = {
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 1  # 矛盾 (Contradictory)
        }
        # 可选：保留原始数据中的额外信息，如类型或qid
        if "qid" in doc_info:
            item["qid"] = doc_info["qid"]
        if "type" in doc_info:
            item["type"] = doc_info["type"]
        elif "contra_type" in doc_info:
            item["type"] = doc_info["contra_type"]

        all_data_list.append(item)

    # 2. 加载所有负例 (Negatives)
    # 标签 0: 非矛盾 (Non-Contradictory)
    for unique_id, doc_info in neg_data.items():
        all_data_list.append({
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 0  # 非矛盾 (Non-Contradictory)
        })

    print(f"\n--- 数据加载统计 ---")
    print(f"正例 (矛盾) 总加载数: {len(pos_data)}")
    print(f"负例 (非矛盾) 总加载数: {len(neg_data)}")
    print(f"评估总样本数: {len(all_data_list)}")

    # 3. 随机打乱并返回
    random.shuffle(all_data_list)
    return all_data_list

# --- 6. 执行脚本 ---

if __name__ == "__main__":

    # 尝试加载数据
    data_path = "sampled_data.json"
    full_data = None
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 数据文件 {data_path} 未找到。评估终止。")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 文件 {data_path} JSON 解析失败。评估终止。")
        sys.exit(1)

    if full_data is None:
        print("错误: 无法加载数据，评估终止。")
        sys.exit(1)

    # 加载所有数据
    contradoc_dataset = load_all_contradoc_data(full_data)

    if len(contradoc_dataset) > 0:
        # 执行评估 (使用 Longformer 单线程推理)
        evaluate_binary_detection_single_threaded(contradoc_dataset)
    else:
        print("警告: 加载的数据集为空，评估未执行。")