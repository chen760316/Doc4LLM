import json
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from typing import Dict, List, Set, Any

# --- 文件路径定义 ---
FILTERED_FILE_PATH = "filtered_dense_text.json"
SAMPLED_FILE_PATH = "../sampled_data.json"
REPORT_FILE_PATH = "binary-report.txt"  # 报告文件路径


def load_json_file(file_path: str) -> Any:
    """安全地加载 JSON 文件并处理常见错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。请检查路径。")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} JSON 解析失败。请检查文件内容。")
        return None


def calculate_metrics_from_ids_custom(filtered_file: str, sampled_file: str):
    """
    根据 ID 的存在性，计算评估指标。
    逻辑映射: 文档 ID 存在于 filtered_data 中 (Retained) -> Predicted Label = 1 (Contradiction)
    所有浮点数结果保留小数点后四位。
    """
    # --- 1. 加载数据 ---
    filtered_data = load_json_file(filtered_file)
    sampled_data = load_json_file(sampled_file)

    if filtered_data is None or sampled_data is None:
        return

    # --- 2. 提取 ID 集合 (修正核心逻辑) ---

    # 2.1 从 sampled_data (ground truth) 中获取 Pos ID 和 Neg ID
    # 修正: 正例 ID 必须基于内部的 "qid" 字段进行提取和集合构建

    pos_qids: Set[str] = set()
    for key, doc_info in sampled_data.get("pos", {}).items():
        if "qid" in doc_info:
            pos_qids.add(doc_info["qid"])
        else:
            # 如果没有 qid，使用外部 key 作为回退，但这可能导致匹配失败
            pos_qids.add(key)

    # 负例 ID 假设它们的外层键就是其唯一 ID (如 QID 或类似)
    neg_ids: Set[str] = set(sampled_data.get("neg", {}).keys())

    all_sampled_ids = pos_qids.union(neg_ids)

    # 2.2 从 filtered_data (predicted results) 中提取所有被识别的 QID
    retained_qids: Set[str] = set()
    if isinstance(filtered_data, list):
        for item in filtered_data:
            # 修正：使用 'source_qid' 字段
            if 'source_qid' in item and item['source_qid']:
                retained_qids.add(item['source_qid'])

    # 最终的 Retained ID 集合 (TP + FP)
    # 我们只评估采样集中的文档，因此只保留出现在 all_sampled_ids 中的被识别 ID
    final_retained_ids = retained_qids.intersection(all_sampled_ids)

    print(f"总样本数 (pos + neg): {len(all_sampled_ids)}")
    print(f"Pos QID 样本数 (Contradiction, True=1): {len(pos_qids)}")
    print(f"Neg ID 样本数 (Non-Contradiction, True=0): {len(neg_ids)}")
    print(f"原始 Retained QID 总数 (来自 {filtered_file}): {len(retained_qids)}")
    print(f"最终用于评估的 Retained ID (Predicted J(D)=1): {len(final_retained_ids)}")

    # --- 3. 统计 TP/FP/FN/TN ---

    # True Positives (TP): 实际是 Pos 且被保留 (Retained)
    tp_count = len(pos_qids.intersection(final_retained_ids))

    # False Positives (FP): 实际是 Neg 却被保留 (Retained)
    # 这里的关键是确保 neg_ids 是唯一ID，且能被 filtered_data 的 qid 匹配（如果 filtered_data 误伤了负例）
    fp_count = len(neg_ids.intersection(final_retained_ids))

    # False Negatives (FN): 实际是 Pos 却被排除 (Not Retained)
    fn_count = len(pos_qids) - tp_count

    # True Negatives (TN): 实际是 Neg 且被排除 (Not Retained)
    tn_count = len(neg_ids) - fp_count

    print("\n--- 详细样本统计 (自定义逻辑) ---")
    print(f"保留的 Pos 样本 (TP): {tp_count}")
    print(f"保留的 Neg 样本 (FP): {fp_count}")
    print(f"排除的 Pos 样本 (FN): {fn_count}")
    print(f"排除的 Neg 样本 (TN): {tn_count}")

    # 交叉检查: TP + FP 应该等于 final_retained_ids 的数量
    if tp_count + fp_count != len(final_retained_ids):
        print(f"!!! 统计错误: TP+FP ({tp_count + fp_count}) 不等于 Final Retained IDs ({len(final_retained_ids)})")
    else:
        print(f"交叉检查: 保留总数 ({tp_count + fp_count}) == 原始统计 ({len(final_retained_ids)})")

    # --- 4. 计算指标所需标签 ---
    binary_true_labels = []
    binary_pred_labels = []

    # 遍历所有采样文档的 ID (Pos QIDs + Neg IDs)
    for doc_id in all_sampled_ids:
        # True label: 1 if in pos_qids (Contradiction), 0 if in neg_ids (Non-Contradiction)
        true_label = 1 if doc_id in pos_qids else 0

        # Predicted label: 1 if in final_retained_ids (Retained), 0 otherwise
        is_retained = doc_id in final_retained_ids

        binary_true_labels.append(true_label)

        # J(D) 预测: Retained -> J(D)=1; Excluded -> J(D)=0
        binary_pred_label = 1 if is_retained else 0
        binary_pred_labels.append(binary_pred_label)

    # --- 5. 最终指标计算 ---
    print("\n--- 评估结果 (Metrics) ---")

    # J(D) 二元判断指标 (标准二元分类)
    binary_accuracy = accuracy_score(binary_true_labels, binary_pred_labels)
    binary_precision = precision_score(binary_true_labels, binary_pred_labels, zero_division=0)
    binary_recall = recall_score(binary_true_labels, binary_pred_labels, zero_division=0)
    binary_f1 = f1_score(binary_true_labels, binary_pred_labels, zero_division=0)

    print("--- 1. J(D) 二元判断指标 (自定义映射分类) ---")
    print(f"Binary Accuracy (二元准确性): {binary_accuracy:.4f}")
    print(f"Binary Precision (二元精确度): {binary_precision:.4f}")
    print(f"Binary Recall (二元召回率): {binary_recall:.4f}")
    print(f"Binary F1 Score (二元 F1 分数): {binary_f1:.4f}")

    # EHR (Evidence Hit Rate)
    # J(D) 预测为 1 (矛盾存在) 的总样本数 (TP + FP)
    predicted_pos_count_binary = len(final_retained_ids)

    # EHR 分子 (TP)
    evidence_hit_on_pos_and_yes = tp_count

    # EHR 计算: TP / (TP + FP) = TP / Predicted Positive Count
    ehr = evidence_hit_on_pos_and_yes / predicted_pos_count_binary if predicted_pos_count_binary > 0 else 0

    print("\n--- 2. EHR (证据命中率) ---")
    print(f"模型预测 '是' (J(D)=1) 的总样本数 (TP+FP): {predicted_pos_count_binary}")
    print(f"V(E) 成功命中数 (来自 Pos 样本, TP): {evidence_hit_on_pos_and_yes}")
    print(f"EHR (Evidence Hit Rate): {ehr:.4f}")

    print("\n--- 3. 详细分类报告 (J(D) 矛盾判断) ---")

    # 手动格式化 classification_report 的输出，保留四位小数
    report_lines = classification_report(
        binary_true_labels,
        binary_pred_labels,
        target_names=["Non-Contradiction (0)", "Contradiction (1)"],
        output_dict=True,
        zero_division=0
    )

    report_str = ""
    report_str += "                       Precision  Recall  F1-Score   Support\n\n"

    for label, metrics in report_lines.items():
        if isinstance(metrics, dict):
            report_str += f"{label:<25} {metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1-score']:.4f}  {metrics['support']:>8}\n"
        elif label == 'accuracy':
            report_str += f"{label:<25}                                 {metrics:.4f}\n"
        elif label in ['macro avg', 'weighted avg']:
            report_str += f"{label:<25} {metrics['precision']:.4f}    {metrics['recall']:.4f}    {metrics['f1-score']:.4f}  {metrics['support']:>8}\n"

    print(report_str)


# --- 主执行部分 ---
if __name__ == "__main__":
    if not os.path.exists(FILTERED_FILE_PATH):
        print(f"\n致命错误: 找不到 {FILTERED_FILE_PATH} 文件。请将文件放置在脚本同一目录下。")
        sys.exit(1)
    # SAMPLED_FILE_PATH 需要上级目录，检查路径是否存在
    if not os.path.exists(SAMPLED_FILE_PATH):
        print(f"\n致命错误: 找不到 {SAMPLED_FILE_PATH} 文件。请检查路径。")
        sys.exit(1)

    # 重定向所有 print 输出到文件
    print(f"正在读取 {FILTERED_FILE_PATH} 和 {SAMPLED_FILE_PATH} 文件...")
    print(f"评估结果将保存到 {REPORT_FILE_PATH} 文件中。")

    try:
        with open(REPORT_FILE_PATH, 'w', encoding='utf-8') as f:
            # 临时保存原始 stdout
            original_stdout = sys.stdout
            sys.stdout = f  # 将 stdout 重定向到文件

            # 打印标题和文件信息
            print("=" * 60)
            print("        二元矛盾判断 (J(D)) 评估报告")
            print("=" * 60)
            print(f"Filtered Data Source: {FILTERED_FILE_PATH}")
            print(f"Sampled Data Source: {SAMPLED_FILE_PATH}")
            print("--- 映射逻辑: ID Retained (source_qid) -> Contradiction (1) ---")

            # 执行评估
            calculate_metrics_from_ids_custom(FILTERED_FILE_PATH, SAMPLED_FILE_PATH)
            print("=" * 60)

    except Exception as e:
        # 确保在出错时也恢复 stdout
        sys.stdout = original_stdout
        print(f"致命错误: 在写入报告文件或执行评估时发生错误: {e}")
        sys.exit(1)

    finally:
        # 恢复 stdout 到控制台
        sys.stdout = original_stdout
        print(f"\n✅ 评估完成。所有结果已成功保存到 {REPORT_FILE_PATH}")