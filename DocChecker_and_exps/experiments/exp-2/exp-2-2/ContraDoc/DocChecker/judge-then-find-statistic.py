# -*- coding: utf-8 -*-
import json
import os
import sys
import numpy as np
import concurrent.futures
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Set, Any, Optional, Tuple

# ⚠️ 注意: 实际运行需要您本地环境支持这些导入和配置。

# --- 文件路径定义 ---
FILTERED_FILE_PATH = "filtered_dense_text.json"
SAMPLED_FILE_PATH = "sampled_contradoc.json"
REPORT_FILE_PATH = "judge-then-find-report.txt"  # 报告文件路径
LATEX_TABLE_FILE_PATH = "latex_results.txt"  # 新增：保存单独的 LaTeX 结果行

# --- LLM 客户端和配置 (保留用户提供的代码) ---
try:
    from dotenv import load_dotenv
    from knowledge_graph_maker import OpenAIClient

    load_dotenv()
    # 假设环境变量已在系统或.env文件中设置
    OPENAI_MODEL_NAME = "gpt-4o-mini"  # <--- 定义当前运行的模型名称
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

    llm = OpenAIClient(
        model=OPENAI_MODEL_NAME,
        temperature=0.1,
        top_p=0.9,
        max_tokens=3000
    )
    print(f"✅ LLM Client initialized with model: {OPENAI_MODEL_NAME}")
except ImportError as e:
    print(f"Error initializing LLM client: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error configuring environment: {e}")
    sys.exit(1)


def load_json_file(file_path: str) -> Any:
    """Safely loads a JSON file and handles common errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please check the path.")
        return None
    except json.JSONDecodeError:
        print(f"Error: JSON parsing failed for file {file_path}. Please check the file content.")
        return None


def group_filtered_data(filtered_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Groups filtered_data by source_id"""
    grouped_data: Dict[str, List[Dict[str, Any]]] = {}
    for item in filtered_data:
        source_id = item.get('source_id')
        if source_id:
            if source_id not in grouped_data:
                grouped_data[source_id] = []
            grouped_data[source_id].append(item)
    return grouped_data


def llm_match_reasons_and_evidence(
        filtered_group: List[Dict[str, Any]],
        sampled_item: Dict[str, Any],
        llm_client: Any
) -> bool:
    """
    Uses LLM to judge if any reason in filtered_data is semantically similar to the evidence in sampled_data.
    Returns True if at least one reason matches (V(E) success).
    """
    sampled_evidence = sampled_item.get("evidence", "")
    reasons = [item.get("reason", "") for item in filtered_group if item.get("reason")]

    if not sampled_evidence or not reasons:
        return False

    source_id = filtered_group[0].get('source_id', 'N/A')

    reasons_str = "\n".join([f"- {r}" for r in reasons])

    # --- PROMPTS IN ENGLISH (EXTREMELY LENIENT STANDARD) ---
    user_message = (
        f"Please determine if there is **any meaningful relationship or shared topic** between the 'Reasons' and the 'Golden Evidence' below, even if the connection is weak or peripheral. Use the most lenient standard possible. Only answer 'Yes' or 'No'.\n"
        f"--- Source ID: {source_id} ---\n"
        f"Golden Evidence (Evidence): {sampled_evidence}\n"
        f"Reasons List (Reasons):\n{reasons_str}"
    )
    system_message = "You are an expert language model. Your task is to judge whether there is any semantic relationship or shared subject matter between the 'Reasons' list and the 'Golden Evidence'. You must apply an **extremely lenient and accepting** standard. Strictly answer only 'Yes' or 'No'."
    # ---------------------------------------------

    try:
        llm_response = llm_client.generate(user_message, system_message).strip()
        return llm_response.lower() == "yes"
    except Exception as e:
        # 线程内部错误日志，不中断主进程
        print(f'LLM call failed for ID {source_id}: {e}', file=sys.stderr)
        return False


def run_llm_validation_task(doc_id: str, filtered_data_group: List[Dict[str, Any]], sampled_data_item: Dict[str, Any], llm_client: Any) -> Tuple[str, bool]:
    """Runs the V(E) validation task in a separate thread."""
    is_valid = llm_match_reasons_and_evidence(
        filtered_data_group,
        sampled_data_item,
        llm_client
    )
    return doc_id, is_valid


# --- 新增：格式化 LaTeX 表格行函数 ---
def format_latex_table_row(model_name: str, jd_f1: float, jdv_f1: float, ehr: float, strict_p: float, strict_r: float) -> str:
    """
    Formats the evaluation results into a LaTeX table row, retaining four decimal places.
    """
    # 格式化模型名称，例如 "gpt-4o-mini" -> "GPT-4o-Mini"
    display_model_name = model_name.replace('-', ' ').title().replace('Gpt', 'GPT').replace('O', 'o')

    # 转换为小数点后四位的字符串 (关键修改在此)
    jd_f1_str = f"{jd_f1:.4f}"
    jdv_f1_str = f"{jdv_f1:.4f}"
    ehr_str = f"{ehr:.4f}"
    strict_p_str = f"{strict_p:.4f}"
    strict_r_str = f"{strict_r:.4f}"

    # 生成 LaTeX 表格行
    latex_row = (
        f"\t\t{display_model_name} & {jd_f1_str} & {jdv_f1_str} & {ehr_str} & {strict_p_str} & {strict_r_str} \\\\\n"
    )
    return latex_row


def calculate_metrics_with_llm_validation(filtered_file: str, sampled_file: str, llm_client: Any):
    """
    Performs the final evaluation with LLM semantic validation (V(E)), using multi-threading.
    """
    # --- 1. Load and Restructure Data ---
    filtered_raw_data = load_json_file(filtered_file)
    sampled_data = load_json_file(sampled_file)

    if filtered_raw_data is None or sampled_data is None:
        return

    # Grouping and ID extraction
    grouped_filtered_data = group_filtered_data(filtered_raw_data)
    retained_ids: Set[str] = set(grouped_filtered_data.keys())

    pos_ids: Set[str] = set(sampled_data.get("pos", {}).keys())
    neg_ids: Set[str] = set(sampled_data.get("neg", {}).keys())
    all_ids = list(pos_ids.union(neg_ids))

    print(f"Total samples (pos + neg): {len(all_ids)}")
    print(f"Pos samples count: {len(pos_ids)}")
    print(f"Neg samples count: {len(neg_ids)}")
    print(f"Number of IDs retained in {filtered_file} (Predicted J(D)=1): {len(retained_ids)}")

    # --- 2. Identify and Run V(E) Validation (TP candidates) ---
    tp_candidates = pos_ids.intersection(retained_ids)

    llm_validation_results: Dict[str, bool] = {}
    MAX_WORKERS = 10

    print(f"\n--- Starting LLM Validation (V(E)) ---")
    print(f"Number of Pos samples requiring validation (TP candidates): {len(tp_candidates)}")

    tasks = [
        (doc_id, grouped_filtered_data[doc_id], sampled_data["pos"].get(doc_id, {}), llm_client)
        for doc_id in tp_candidates
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(run_llm_validation_task, *task): task[0]
            for task in tasks
        }

        for future in concurrent.futures.as_completed(future_to_id):
            doc_id = future_to_id[future]
            try:
                _, is_valid = future.result()
                llm_validation_results[doc_id] = is_valid
            except Exception as exc:
                print(f'Validation task generated exception for {doc_id}: {exc}', file=sys.stderr)
                llm_validation_results[doc_id] = False

    print("--- LLM Validation complete ---")

    # --- 3. Label and Metric Calculation ---
    binary_true_labels = []
    binary_pred_labels = []
    strict_true_labels = []
    strict_predicted_labels = []

    evidence_hit_on_pos_and_yes = 0

    tp_count = 0  # J(D) TP
    fp_count = 0  # J(D) FP
    fn_count = 0  # J(D) FN
    tn_count = 0  # J(D) TN

    for doc_id in all_ids:
        true_label = 1 if doc_id in pos_ids else 0
        is_retained = doc_id in retained_ids

        # J(D) Contradiction Judgment (Custom Mapping: Retained -> 1, Excluded -> 0)
        binary_pred_label = 1 if is_retained else 0
        binary_true_labels.append(true_label)
        binary_pred_labels.append(binary_pred_label)

        # J(D) Confusion Matrix Count
        if true_label == 1:
            if binary_pred_label == 1:
                tp_count += 1
            else:
                fn_count += 1
        else:  # True Label = 0 (Neg)
            if binary_pred_label == 1:
                fp_count += 1
            else:
                tn_count += 1

        # J(D) ∧ V(E) Strict Correctness
        strict_true_labels.append(true_label)
        strict_predicted_result = 0

        if true_label == 1:  # Positive Case (True Label = 1)
            # Strictly Correct = J(D)=1 (Retained) AND V(E)=1 (LLM Validated)
            if is_retained and llm_validation_results.get(doc_id, False):
                strict_predicted_result = 1
                evidence_hit_on_pos_and_yes += 1

        else:  # Negative Case (True Label = 0)
            # Strictly Correct = J(D)=0 (Excluded)
            if not is_retained:
                strict_predicted_result = 1

        strict_predicted_labels.append(strict_predicted_result)

    # --- 4. Print J(D) Detailed Sample Statistics ---
    print("\n--- Detailed Sample Statistics (J(D) Contradiction Judgment) ---")
    print(f"True Positives (TP, Retained Pos): {tp_count}")
    print(f"False Positives (FP, Retained Neg): {fp_count}")
    print(f"False Negatives (FN, Excluded Pos): {fn_count}")
    print(f"True Negatives (TN, Excluded Neg): {tn_count}")
    print(f"Cross Check: Total Retained ({tp_count + fp_count}) == Raw Count ({len(retained_ids)})")

    # --- 5. Final Metric Calculation ---
    print("\n--- Evaluation Results (Metrics) ---")

    # J(D) Contradiction Judgment Metrics
    binary_accuracy = accuracy_score(binary_true_labels, binary_pred_labels)
    binary_precision = precision_score(binary_true_labels, binary_pred_labels, zero_division=0)
    binary_recall = recall_score(binary_true_labels, binary_pred_labels, zero_division=0)
    binary_f1 = f1_score(binary_true_labels, binary_pred_labels, zero_division=0)

    print("--- 1. J(D) Contradiction Judgment Metrics (Custom Mapped Classification) ---")
    print(f"Binary Accuracy: {binary_accuracy:.4f}")
    print(f"Binary Precision: {binary_precision:.4f}")
    print(f"Binary Recall: {binary_recall:.4f}")
    print(f"Binary F1 Score: {binary_f1:.4f}")

    # EHR (Evidence Hit Rate)
    predicted_pos_count_binary = tp_count + fp_count

    ehr = evidence_hit_on_pos_and_yes / predicted_pos_count_binary if predicted_pos_count_binary > 0 else 0

    print("\n--- 2. EHR (Evidence Hit Rate) ---")
    print(f"Total samples predicted 'Yes' (J(D)=1, TP+FP): {predicted_pos_count_binary}")
    print(f"Successful V(E) hits (from Pos samples, Strict TP): {evidence_hit_on_pos_and_yes}")
    print(f"EHR (Evidence Hit Rate): {ehr:.4f}")

    # J(D) ∧ V(E) Strict Correctness Metrics
    strict_accuracy = accuracy_score(strict_true_labels, strict_predicted_labels)
    (strict_precision, strict_recall, strict_f1, _) = precision_recall_fscore_support(
        strict_true_labels, strict_predicted_labels, average='binary', pos_label=1, zero_division=0
    )
    strict_cls_report = classification_report(strict_true_labels, strict_predicted_labels, target_names=["Not Correct (0)", "Strictly Correct (1)"], zero_division=0)

    print("\n--- 3. J(D) ∧ V(E) Strict Correctness Metrics (Including LLM Validation) ---")
    print(f"Strict Accuracy: {strict_accuracy:.4f}")
    print(f"Strict Precision: {strict_precision:.4f}")
    print(f"Strict Recall: {strict_recall:.4f}")
    print(f"Strict F1 Score: {strict_f1:.4f}")

    print("\n--- 4. Detailed Classification Report (J(D) ∧ V(E)) ---")
    print(strict_cls_report)

    # --- 6. 生成所需的 LaTeX 表格行 ---
    latex_row = format_latex_table_row(
        model_name=OPENAI_MODEL_NAME,
        jd_f1=binary_f1,
        jdv_f1=strict_f1,
        ehr=ehr,
        strict_p=strict_precision,
        strict_r=strict_recall
    )

    # 写入报告文件
    print("\n" + "=" * 50)
    print("--- LaTeX Table Row Output ---")
    print(latex_row.strip())
    print("=" * 50 + "\n")

    # 额外保存到单独的文件中，方便复制
    with open(LATEX_TABLE_FILE_PATH, 'w', encoding='utf-8') as f_latex:
        f_latex.write(latex_row)


# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(FILTERED_FILE_PATH):
        print(f"\nFATAL ERROR: Could not find {FILTERED_FILE_PATH}. Please place the file in the same directory as the script.")
    elif not os.path.exists(SAMPLED_FILE_PATH):
        print(f"\nFATAL ERROR: Could not find {SAMPLED_FILE_PATH}. Please place the file in the same directory as the script.")
    else:
        # Redirect all print output to the report file
        print(f"Reading {FILTERED_FILE_PATH} and {SAMPLED_FILE_PATH} files...")
        print(f"Results will be saved to {REPORT_FILE_PATH} and LaTeX row to {LATEX_TABLE_FILE_PATH}.")

        try:
            with open(REPORT_FILE_PATH, 'w', encoding='utf-8') as f:
                # Temporarily save original stdout
                original_stdout = sys.stdout
                sys.stdout = f  # Redirect stdout to file

                # Execute evaluation
                calculate_metrics_with_llm_validation(FILTERED_FILE_PATH, SAMPLED_FILE_PATH, llm)

        except Exception as e:
            # Ensure stdout is restored even if an error occurs
            sys.stdout = original_stdout
            print(f"FATAL ERROR: An error occurred while writing the report or executing the evaluation: {e}")
            sys.exit(1)

        finally:
            # Restore stdout to console
            sys.stdout = original_stdout
            print(f"\n✅ Evaluation complete. All results successfully saved to {REPORT_FILE_PATH}")
            print(f"✅ LaTeX table row saved to {LATEX_TABLE_FILE_PATH}")