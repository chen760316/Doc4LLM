# -*- coding: utf-8 -*-
import json
import os
import random
from typing import Dict, List, Any, Tuple
import collections
# 移除了 threading 和 concurrent.futures，因为 Longformer 推理通常使用 PyTorch/TensorFlow 的 DataLoader 进行批量处理，而不是 Python 线程池
import time

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
    exit()

# --- 1. 初始化和配置 (使用 Longformer 配置) ---
# 移除了所有 API 密钥和客户端初始化代码

# Longformer 模型配置
# ⚠️ 替换此处为您微调过的模型路径或一个通用的 Longformer 基座模型
DEFAULT_HF_MODEL = "allenai/longformer-base-4096"

print("--- 1. 正在加载本地 Longformer 模型 ---")
try:
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_HF_MODEL)
    # 此处加载的模型是用于序列分类 (Sequence Classification) 的 Longformer
    # 注意：使用基模型进行零样本分类的性能可能很差，建议加载微调过的模型
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_HF_MODEL, num_labels=2)
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"✅ Longformer 模型 ({DEFAULT_HF_MODEL}) 加载成功，设备: {device}")
except Exception as e:
    print(f"❌ 错误: Longformer 模型加载失败: {e}")
    model = None


# --- 2. 核心检测函数 (替换为 Longformer 推理) ---

def predict_conflict_with_longformer(document_text: str) -> Tuple[int, str]:
    """
    使用 Longformer 模型进行二元分类推理。
    标签: 1 for YES (矛盾), 0 for NO (非矛盾)
    解释: 本地模型不提供生成解释的能力，返回分类状态。
    """
    if model is None:
        return 0, "模型未加载或初始化失败。"

    try:
        # 1. 编码输入
        # Longformer 最大长度通常为 4096，这里进行截断和填充
        inputs = tokenizer(
            document_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True  # 动态填充 (更高效)，但在单例循环中影响不大
        ).to(device)

        # 2. 推理
        with torch.no_grad():
            outputs = model(**inputs)
            # 获取 logits
            logits = outputs.logits
            # 预测类别 (0 或 1)
            predicted_class_id = torch.argmax(logits, dim=1).item()

        # 3. 结果映射
        label = predicted_class_id

        # 结果解释（仅为状态描述）
        if label == 1:
            explanation = "Longformer 模型分类为矛盾 (Contradictory)."
        else:
            explanation = "Longformer 模型分类为非矛盾 (Non-Contradictory)."

        return label, explanation

    except Exception as e:
        return 0, f"Longformer 推理过程中发生异常: {e}"


# --- 3. 模拟/加载数据集函数 (保持不变) ---

def load_and_sample_contradoc_data(full_data: Dict, num_pos: int = 30, num_neg: int = 30) -> List[Dict[str, Any]]:
    """从 ContraDoc JSON 结构中加载数据，并按照要求进行抽样。（保持不变）"""
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    # ... (此处省略抽样逻辑代码，保持与您原代码一致) ...
    pos_list = []
    pos_grouped = collections.defaultdict(list)
    for unique_id, doc_info in pos_data.items():
        types = doc_info.get("contra_type", ["Unknown"])
        for c_type in types:
            pos_grouped[c_type].append({
                "unique_id": unique_id,
                "text": doc_info["text"],
                "label": 1,
                "type": c_type
            })

    all_types = list(pos_grouped.keys())
    base_sample_per_type = num_pos // len(all_types)
    remaining_to_sample = num_pos
    sampled_ids = set()

    for c_type in all_types:
        num_to_sample = min(base_sample_per_type, len(pos_grouped[c_type]))
        available_docs = [d for d in pos_grouped[c_type] if d['unique_id'] not in sampled_ids]
        if available_docs:
            sample = random.sample(available_docs, min(num_to_sample, len(available_docs)))
            for doc in sample:
                if doc['unique_id'] not in sampled_ids:
                    pos_list.append(doc)
                    sampled_ids.add(doc['unique_id'])
                    remaining_to_sample -= 1

    if remaining_to_sample > 0:
        all_remaining_docs = []
        for c_type in all_types:
            available_docs = [d for d in pos_grouped[c_type] if d['unique_id'] not in sampled_ids]
            all_remaining_docs.extend(available_docs)

        unique_remaining_docs = list({d['unique_id']: d for d in all_remaining_docs}.values())

        if unique_remaining_docs:
            final_sample = random.sample(unique_remaining_docs, min(remaining_to_sample, len(unique_remaining_docs)))
            pos_list.extend(final_sample)

    pos_list = pos_list[:num_pos]

    # 2. 处理负例 (Negatives)
    neg_available = []
    for unique_id, doc_info in neg_data.items():
        neg_available.append({
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 0
        })

    num_to_sample_neg = min(num_neg, len(neg_available))
    neg_list = random.sample(neg_available, num_to_sample_neg)

    print(f"\n--- 数据抽样统计 ---")
    print(f"正例 (矛盾) 抽取数: {len(pos_list)} / 目标 {num_pos}")
    print(f"负例 (非矛盾) 抽取数: {len(neg_list)} / 目标 {num_neg}")

    # 3. 合并和返回
    dataset = pos_list + neg_list
    random.shuffle(dataset)
    return dataset


# --- 4. 结果保存辅助函数 (修改保存目录和模型名) ---

def save_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存评估指标和详细的文档结果到本地文件，保存至 report/binary_judge_report_longformer 目录下。"""

    # 定义目标目录
    SAVE_DIR = "report/binary_judge_report_longformer"  # <--- 修改目录
    # 创建目录（如果不存在）
    os.makedirs(SAVE_DIR, exist_ok=True)

    base_filename = f"ContraDoc_Binary_Eval_{model_name.replace('/', '_')}"  # 替换模型名中的斜杠

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
        # 写入核心指标
        f.write(f"Accuracy (准确性): {metrics.get('Accuracy', 'N/A')}\n")
        f.write(f"Precision (精确度): {metrics.get('Precision', 'N/A')}\n")
        f.write(f"Recall (召回率): {metrics.get('Recall', 'N/A')}\n")
        f.write(f"F1 Score (F1 分数): {metrics.get('F1_Score', 'N/A')}\n")
        f.write("-" * 40 + "\n")
        f.write("详细分类报告:\n")
        f.write(metrics.get("classification_report", "N/A"))

    print(f"✨ 最终评估指标已保存到: {txt_path}")


# --- 5. 评估主函数 (修改为 Longformer 评估，使用单线程) ---

def evaluate_binary_detection_single_threaded(dataset: List[Dict[str, Any]]):
    """
    对数据集执行二元检测并计算评估指标，使用单线程 (适合本地模型推理)。
    """
    true_labels = []
    pred_labels = []
    detailed_results = []

    model_name = DEFAULT_HF_MODEL  # 使用 Longformer 模型名
    total_samples = len(dataset)
    print(f"\n--- 开始评估 ({total_samples} 个文档, 模型: {model_name}, 使用单线程/本地推理) ---")

    start_time = time.time()

    for processed_count, item in enumerate(dataset, 1):
        doc_text = item["text"]
        unique_id = item["unique_id"]
        true_label = item["label"]

        # 调用 Longformer 推理函数 (不再需要 client 和 system_prompt)
        predicted_label, explanation = predict_conflict_with_longformer(doc_text)

        # 收集结果
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

        # 打印进度 (每 10 个打印一次，或在结束时打印)
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
    # ... (指标计算逻辑保持不变) ...

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


# --- 6. 执行脚本 ---

if __name__ == "__main__":

    # 尝试加载数据 (保持不变)
    try:
        data_path = "../../../../ContraDOC/sampled_contradoc.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        try:
            data_path = "../../../../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"
            with open(data_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: ContraDoc.json not found. Please check paths.")
            exit()
        except json.JSONDecodeError:
            print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
            exit()
    except json.JSONDecodeError:
        print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
        exit()

    if full_data is None:
        print("错误: 无法加载 ContraDoc 数据，评估终止。")
        exit()

    # 使用抽样函数加载数据集 (默认 30 正例, 30 负例)
    contradoc_dataset = load_and_sample_contradoc_data(full_data, num_pos=30, num_neg=30)

    if len(contradoc_dataset) > 0:
        # 执行评估 (使用 Longformer 单线程推理)
        evaluate_binary_detection_single_threaded(contradoc_dataset)
    else:
        print("警告: 加载的数据集为空，评估未执行。")