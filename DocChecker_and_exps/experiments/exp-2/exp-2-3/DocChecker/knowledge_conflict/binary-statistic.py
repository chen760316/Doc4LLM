import json
import os
import sys
import numpy as np
from typing import Set, Any, Tuple, Optional, List

# --- 文件路径定义 ---
FILTERED_FILE_PATH = "filtered_dense_text.json"
SAMPLED_FILE_PATH = "sampled_data.json"  # True Positive (Label=1) 样本列表
REPORT_FILE_PATH = "binary-report.txt"  # 报告文件路径
INDEX_KEY = "source_index"  # filtered_dense_text.json 中对应列表索引的字段


def load_json_file(file_path: str) -> Any:
    """安全地加载 JSON 文件并处理常见错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"警告: {file_path} 预期为列表结构，但加载为 {type(data)}。可能导致错误。")
            return data
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}。请检查路径。")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} JSON 解析失败。请检查文件内容。")
        return None


def get_true_positive_indices(data: List[Any]) -> int:
    """
    获取 sampled_data.json 中 True Positive 样本的总数 (即列表长度)。
    """
    if isinstance(data, list):
        return len(data)
    return 0


def get_predicted_positive_indices(data: List[Any]) -> Set[str]:
    """
    从 filtered_dense_text.json 中提取被模型保留的样本的索引 (Predicted Positives)。
    索引被转换为字符串以进行集合操作。
    """
    retained_indices: Set[str] = set()
    if isinstance(data, list):
        for item in data:
            if INDEX_KEY in item:
                # 提取 source_index (作为字符串)
                retained_indices.add(str(item[INDEX_KEY]))
    return retained_indices


def calculate_metrics_from_indices_matching(filtered_file: str, sampled_file: str):
    """
    根据列表索引进行匹配，计算指标。
    """
    # --- 1. 加载数据 ---
    filtered_data = load_json_file(filtered_file)
    sampled_data = load_json_file(sampled_file)

    if filtered_data is None or sampled_data is None:
        return

    # --- 2. 提取 True/Predicted Positives ---

    # N_true_pos (真实正例总数): sampled_data 的长度
    n_true_pos = get_true_positive_indices(sampled_data)

    # Predicted Positives Indices: filtered_data 中的 source_index 集合
    retained_indices_pred: Set[str] = get_predicted_positive_indices(filtered_data)

    # 真实的 Positive 索引集合 (0 到 N_true_pos - 1)
    # 必须是字符串类型才能与 retained_indices_pred 集合进行交集操作
    pos_indices_true: Set[str] = {str(i) for i in range(n_true_pos)}

    # N_pred_pos (预测正例总数): retained_indices_pred 的大小
    n_pred_pos = len(retained_indices_pred)

    if n_true_pos == 0:
        print("错误: sampled_data 中没有 True Positives 样本。无法计算指标。")
        return

    # --- 3. 统计 TP/FP/FN ---

    # TP: True Positives (真正例): 实际是 Pos 且被保留 (Retained)
    tp_count = len(pos_indices_true.intersection(retained_indices_pred))

    # FN: False Negatives (假负例): 实际是 Pos 且被排除 (Excluded)
    fn_count = n_true_pos - tp_count

    # FP: False Positives (假正例): 被保留但不是 True Positives
    # (即，Retained 且不在 sampled_data 索引范围内的部分)
    fp_count = n_pred_pos - tp_count

    # TN: True Negatives 无法计算
    tn_count = 0

    # Total evaluated samples (用于计算 Accuracy 分母的估算值)
    n_total_evaluated = n_true_pos + fp_count

    print(f"--- 数据概览 ---")
    print(f"True Positive 样本总数 (N_true_pos, sampled_data 长度): {n_true_pos}")
    print(f"Predicted Positive 样本总数 (N_pred_pos, filtered_data 长度): {n_pred_pos}")

    print("\n--- 详细样本统计 ---")
    print(f"True Positives (TP): {tp_count}")
    print(f"False Negatives (FN): {fn_count}")
    print(f"False Positives (FP, 估算): {fp_count}")
    print(f"True Negatives (TN): 0 (无法计算)")

    # --- 4. 最终指标计算 ---
    print("\n--- 评估结果 ---")

    # 召回率 (Recall):
    binary_recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0

    # 精确度 (Precision): (TP / Predicted Positives)
    binary_precision = tp_count / n_pred_pos if n_pred_pos > 0 else 0

    # F1 Score:
    binary_f1 = (2 * binary_precision * binary_recall) / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0

    # 准确率 (Accuracy): 估算值 TP / (TP + FN + FP)
    binary_accuracy_est = tp_count / n_total_evaluated if n_total_evaluated > 0 else 0

    print(f"召回率 (Recall): {binary_recall:.4f}")
    print(f"精确度 (Precision): {binary_precision:.4f}")
    print(f"F1 Score: {binary_f1:.4f}")
    print(f"准确率 (Accuracy, 估算值): {binary_accuracy_est:.4f} (⚠️ 仅估算，因缺少负例)")
    print("-" * 30)
    print(f"EHR (证据命中率) = 精确度 = {binary_precision:.4f}")


# --- 主执行部分 ---
if __name__ == "__main__":

    if not os.path.exists(FILTERED_FILE_PATH):
        print(f"\n致命错误: 找不到 {FILTERED_FILE_PATH} 文件。请将文件放置在脚本同一目录下。")
    elif not os.path.exists(SAMPLED_FILE_PATH):
        print(f"\n致命错误: 找不到 {SAMPLED_FILE_PATH} 文件。请将文件放置在脚本同一目录下。")
    else:
        # 重定向所有 print 输出到文件
        print(f"正在读取 {FILTERED_FILE_PATH} (Predicted Pos) 和 {SAMPLED_FILE_PATH} (True Pos) 文件...")
        print(f"评估结果将保存到 {REPORT_FILE_PATH} 文件中。")

        original_stdout = sys.stdout
        try:
            with open(REPORT_FILE_PATH, 'w', encoding='utf-8') as f:
                sys.stdout = f

                # 执行评估
                calculate_metrics_from_indices_matching(FILTERED_FILE_PATH, SAMPLED_FILE_PATH)

        except Exception as e:
            sys.stdout = original_stdout
            print(f"致命错误: 在写入报告文件或执行评估时发生错误: {e}")
            sys.exit(1)

        finally:
            sys.stdout = original_stdout
            print(f"\n✅ 评估完成。所有结果已成功保存到 {REPORT_FILE_PATH}")