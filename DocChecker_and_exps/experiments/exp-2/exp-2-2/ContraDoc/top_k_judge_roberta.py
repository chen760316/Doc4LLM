# -*- coding: utf-8 -*-
import json
from dotenv import load_dotenv
import os
import random
from typing import Dict, List, Any, Tuple, Union
from datetime import datetime
import time
import sys

# --- 引入 Hugging Face 库 ---
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
    import numpy as np  # 用于处理批处理结果

    # 尝试使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("❌ 错误: 缺少必要的库 (transformers, torch, 或 sklearn)。请运行 pip install transformers torch sklearn")
    sys.exit(1)

# --- 1. 配置和 RoBERTa NLI 模型初始化 ---
# 使用 RoBERTa-large-mnli 进行二元矛盾检测
MODEL_NAME = "roberta-large-mnli"
MAX_LENGTH = 512  # RoBERTa 的最大输入长度限制
SAVE_DIR = "report/binary_judge_report_roberta_nli"  # 新的保存目录

print(f"--- 1. 正在加载 Hugging Face NLI 模型 ({MODEL_NAME}) ---")
print(f"!!! 警告: 模型最大输入长度为 {MAX_LENGTH}，长文本将被截断。")
print(f"✅ 使用设备: {device}")

# 全局变量用于存储模型和分词器
tokenizer = None
model = None
id2label = {}

try:
    # 自动加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # 强制将模型移到 GPU
    model.to(device)
    model.eval()
    print(f"✅ NLI 模型加载成功")

    # 获取模型的标签数量和ID到标签名的映射
    id2label = model.config.id2label
    print(f"✅ 模型标签映射: {id2label}")
    if len(id2label) != 3:
        raise ValueError("NLI 模型预期有 3 个标签 (contradiction, entailment, neutral)。")

except Exception as e:
    print(f"❌ 错误: 模型加载失败: {e}")
    sys.exit(1)


# --- 2. 核心检测函数 (NLI 3标签 -> 2标签映射) ---

def predict_conflict_with_roberta_nli(document_text: Union[str, List[str]]) -> Tuple[Union[int, List[int]], str]:
    """
    使用 NLI 模型进行分类推理，支持单文本或批量推理。
    将 NLI 3 个标签中的 "Contradiction" 映射为 1 (矛盾)，其他映射为 0 (非矛盾)。
    """
    is_batch = isinstance(document_text, list)

    if model is None:
        explanation = "模型未加载或初始化失败。"
        return ([0] * len(document_text) if is_batch else 0), explanation

    try:
        # 1. 编码输入
        inputs = tokenizer(
            document_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # 预测类别 ID
            predicted_class_ids = torch.argmax(logits, dim=1).cpu().numpy()

        # --- 标签映射 ---
        # 找到 'contradiction' 对应的 ID
        contra_id = -1
        for id, label in id2label.items():
            if 'contradiction' in label.lower():
                contra_id = id
                break

        if contra_id == -1:
            raise ValueError("模型配置中找不到 'contradiction' 标签ID。")

        # 映射到二元输出 (1=Contradictory, 0=Non-Contradictory)
        # 如果预测 ID 等于矛盾 ID，则标签为 1，否则为 0
        binary_labels = np.where(predicted_class_ids == contra_id, 1, 0).tolist()

        # -------------------------------

        if is_batch:
            explanation = f"NLI 模型批量分类结果，映射为二元标签。"
            return binary_labels, explanation
        else:
            predicted_label_name = id2label.get(predicted_class_ids[0], 'UNKNOWN')
            explanation = f"NLI 模型分类结果: {predicted_label_name} (ID: {predicted_class_ids[0]}). 二元映射为 {'矛盾 (1)' if binary_labels[0] == 1 else '非矛盾 (0)'}."
            return binary_labels[0], explanation

    except Exception as e:
        error_msg = f"NLI 推理过程中发生异常: {e}"
        if is_batch:
            return ([0] * len(document_text)), error_msg
        else:
            return 0, error_msg


# --- 3. 数据集加载函数 (加载 Positives 和 Negatives) ---

def load_all_contradoc_data(full_data: Dict) -> List[Dict[str, Any]]:
    """
    从 ContraDoc JSON 结构中加载所有正例 (pos) 和负例 (neg) 数据。
    """
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    all_data_list = []

    # 1. 加载所有正例 (Positives) -> 标签 1: 矛盾 (Contradictory)
    for unique_id, doc_info in pos_data.items():
        item = {
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 1
        }
        all_data_list.append(item)

    # 2. 加载所有负例 (Negatives) -> 标签 0: 非矛盾 (Non-Contradictory)
    for unique_id, doc_info in neg_data.items():
        all_data_list.append({
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 0
        })

    print(f"\n--- 数据加载统计 ---")
    print(f"正例 (矛盾, 标签=1) 总加载数: {len(pos_data)}")
    print(f"负例 (非矛盾, 标签=0) 总加载数: {len(neg_data)}")
    print(f"评估总样本数: {len(all_data_list)}")

    # 3. 随机打乱并返回
    random.shuffle(all_data_list)
    return all_data_list


# --- 4. 评估主函数 (批量推理) ---

def evaluate_binary_detection_batched(dataset: List[Dict[str, Any]], batch_size: int = 32):
    """
    对数据集执行二元检测并计算评估指标，使用批量推理。
    """
    true_labels = []
    pred_labels = []
    detailed_results = []

    model_name = MODEL_NAME
    total_samples = len(dataset)
    print(f"\n--- 开始评估 ({total_samples} 个文档, 模型: {model_name}, 使用批量推理, Batch Size: {batch_size}) ---")

    start_time = time.time()

    # 将数据集分批处理
    for i in range(0, total_samples, batch_size):
        batch = dataset[i:i + batch_size]
        batch_texts = [item['text'] for item in batch]
        batch_true_labels = [item['label'] for item in batch]

        # 批量预测
        predicted_labels_batch, explanation = predict_conflict_with_roberta_nli(batch_texts)

        # 更新结果
        for j, item in enumerate(batch):
            true_label = batch_true_labels[j]
            predicted_label = predicted_labels_batch[j]

            true_labels.append(true_label)
            pred_labels.append(predicted_label)

            result_entry = {
                "unique_id": item["unique_id"],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "explanation": explanation if j == 0 else "批量结果",  # 仅在第一个记录中记录完整的解释
                "is_correct": (true_label == predicted_label)
            }
            detailed_results.append(result_entry)

        processed_count = i + len(batch)
        if processed_count % (batch_size * 2) == 0 or processed_count == total_samples:
            print(f"✅ 已处理 {processed_count}/{total_samples} 个文档... 耗时: {time.time() - start_time:.2f}s")

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
        "Total_Samples": total_samples,
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1_Score": f"{f1:.4f}",
        "classification_report": cls_report
    }

    save_results_to_file(metrics_to_save, detailed_results, model_name)


# --- 5. 结果保存辅助函数 (已更新为 RoBERTa NLI 路径) ---

def save_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存评估指标和详细的文档结果到本地文件。"""

    os.makedirs(SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ContraDoc_Binary_Eval_{model_name.replace('/', '_')}_{timestamp}"

    # 1. 保存详细结果 (JSON)
    json_path = os.path.join(SAVE_DIR, f"{base_filename}_details.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"\n✨ 详细评估结果已保存到: {json_path}")

    # 2. 保存指标结果 (TXT)
    txt_path = os.path.join(SAVE_DIR, f"{base_filename}_metrics.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ContraDoc 二元检测评估指标 ({model_name}) ---\n")
        f.write(f"时间: {timestamp}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy (准确性): {metrics.get('Accuracy', 'N/A')}\n")
        f.write(f"Precision (精确度): {metrics.get('Precision', 'N/A')}\n")
        f.write(f"Recall (召回率): {metrics.get('Recall', 'N/A')}\n")
        f.write(f"F1 Score (F1 分数): {metrics.get('F1_Score', 'N/A')}\n")
        f.write("-" * 40 + "\n")
        f.write("详细分类报告:\n")
        f.write(metrics.get("classification_report", "N/A"))

    print(f"✨ 最终评估指标已保存到: {txt_path}")


# --- 6. 执行脚本 ---

if __name__ == "__main__":

    # 尝试加载数据
    data_path = "../../../../ContraDOC/sampled_contradoc.json"
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

    # 加载所有数据 (正例和负例)
    contradoc_dataset = load_all_contradoc_data(full_data)

    # 定义批次大小
    BATCH_SIZE = 32

    if len(contradoc_dataset) > 0:
        # 执行评估 (使用 RoBERTa NLI 批量推理)
        evaluate_binary_detection_batched(contradoc_dataset, batch_size=BATCH_SIZE)
    else:
        print("警告: 加载的数据集为空，评估未执行。")