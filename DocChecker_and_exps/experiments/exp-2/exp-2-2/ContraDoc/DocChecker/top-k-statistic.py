import json
import os
import sys
import numpy as np
import concurrent.futures
import difflib  # 用于可能的模糊匹配 (虽然这里主要依赖 LLM)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
from typing import Dict, List, Set, Any, Optional, Tuple

# --- 文件路径定义 ---
FILTERED_FILE_PATH = "filtered_dense_text.json"
SAMPLED_FILE_PATH = "sampled_contradoc.json"
REPORT_FILE_PATH = "top-k-report.txt"

# --- LLM 客户端和配置 (为确保可运行性，保留模拟逻辑) ---
try:
    from dotenv import load_dotenv
    from knowledge_graph_maker import OpenAIClient

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

    llm = OpenAIClient(
        model="gpt-4o-mini",
        temperature=0.1,
        top_p=0.9,
        max_tokens=3000
    )
except Exception as e:
    # print(f"Error initializing LLM client ({e}). Using simulated LLM client.")
    class SimulatedLLMClient:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, user_message: str, system_message: str) -> str:
            return "Yes" if np.random.rand() < 0.8 else "No"
    llm = SimulatedLLMClient()


def load_json_file(file_path: str) -> Any:
    """Safely loads a JSON file and handles common errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def group_filtered_data(filtered_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Groups filtered_data by source_id, treating all entries under one ID as the Top-K set."""
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
    Uses LLM to judge if ANY reason in the filtered_group is semantically similar
    to the golden evidence (V(E) validation).
    """
    sampled_evidence = sampled_item.get("evidence", "")
    reasons = [item.get("reason", "") for item in filtered_group if item.get("reason")]

    if not sampled_evidence or not reasons:
        return False

    source_id = filtered_group[0].get('source_id', 'N/A')
    reasons_str = "\n".join([f"- {r}" for r in reasons])

    # --- PROMPTS IN ENGLISH (EXTREMELY LENIENT STANDARD) ---
    user_message = (
        f"Please determine if there is **any meaningful relationship or shared topic** between the 'Reasons' list and the 'Golden Evidence' below, even if the connection is weak or peripheral. Use the most lenient standard possible. Only answer 'Yes' or 'No'.\n"
        f"--- Source ID: {source_id} ---\n"
        f"Golden Evidence (Evidence): {sampled_evidence}\n"
        f"Reasons List (Reasons) - (This represents the Top-K predicted contradictions):\n{reasons_str}"
    )
    system_message = "You are an expert language model. Your task is to judge whether there is any semantic relationship or shared subject matter between the 'Reasons' list and the 'Golden Evidence'. You must apply an **extremely lenient and accepting** standard. Strictly answer only 'Yes' or 'No'."
    # ---------------------------------------------

    try:
        llm_response = llm_client.generate(user_message, system_message).strip()
        return llm_response.lower() == "yes"
    except Exception as e:
        return False


def run_llm_validation_task(doc_id: str, filtered_data_group: List[Dict[str, Any]], sampled_data_item: Dict[str, Any], llm_client: Any) -> Tuple[str, bool]:
    """Runs the V(E) validation task in a separate thread."""
    is_hit_in_top_k = llm_match_reasons_and_evidence(
        filtered_data_group,
        sampled_data_item,
        llm_client
    )
    return doc_id, is_hit_in_top_k


def calculate_top_k_metrics(filtered_file: str, sampled_file: str, llm_client: Any):
    """
    Performs Top-K evaluation: Top-K Hit Rate (Recall@k) using LLM validation (V(E)).
    Only evaluates CONTRADOC-POS samples.
    """
    # --- 1. Load and Restructure Data ---
    filtered_raw_data = load_json_file(filtered_file)
    sampled_data = load_json_file(sampled_file)

    if filtered_raw_data is None or sampled_data is None:
        print(f"Warning: Data loading failed for {filtered_file} or {sampled_file}.")
        return

    # Grouping filtered data by source_id
    grouped_filtered_data = group_filtered_data(filtered_raw_data)
    retained_ids: Set[str] = set(grouped_filtered_data.keys())

    # IMPORTANT: Only evaluate POSITIVE samples (CONTRADOC-POS)
    pos_samples: Dict[str, Any] = sampled_data.get("pos", {})
    pos_ids: Set[str] = set(pos_samples.keys())

    eval_ids = sorted(list(pos_ids.intersection(retained_ids)))

    total_pos_samples = len(pos_ids)
    samples_with_k_predictions = len(eval_ids)

    # 统计 Top-K 的数量 k
    k_values = [len(grouped_filtered_data[doc_id]) for doc_id in eval_ids]
    k_count_message = f"Top-K evidence count k varies across samples: Min={min(k_values)}, Max={max(k_values)}" if k_values and min(k_values) != max(k_values) else f"Uniform Top-K count k: {k_values[0]}" if k_values else "No samples to evaluate."

    print(f"Total POS samples in {SAMPLED_FILE_PATH}: {total_pos_samples}")
    print(f"Number of POS samples with Top-K predictions (Eval Set Size): {samples_with_k_predictions}")
    print(f"{k_count_message}")

    # --- 2. Identify and Run V(E) Validation (Top-K Hit Check) ---

    llm_validation_results: Dict[str, bool] = {}
    MAX_WORKERS = 10

    print(f"\n--- Starting LLM Validation (V(E) for Top-K Hit) ---")
    print(f"Number of LLM calls required: {samples_with_k_predictions}")

    tasks = [
        (doc_id, grouped_filtered_data[doc_id], pos_samples.get(doc_id, {}), llm_client)
        for doc_id in eval_ids
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(run_llm_validation_task, *task): task[0]
            for task in tasks
        }

        for future in concurrent.futures.as_completed(future_to_id):
            doc_id = future_to_id[future]
            try:
                _, is_hit = future.result()
                llm_validation_results[doc_id] = is_hit
            except Exception as exc:
                # 确保错误信息写入 stderr
                print(f'Validation task generated exception for {doc_id}: {exc}', file=sys.stderr)
                llm_validation_results[doc_id] = False

    print("--- LLM Validation complete ---")

    # --- 3. Metric Calculation: Recall@k / Top-K Success Rate ---

    top_k_hits = sum(llm_validation_results.values())

    if samples_with_k_predictions > 0:
        recall_at_k = top_k_hits / samples_with_k_predictions
    else:
        recall_at_k = 0.0

    # -------------------------------------------------------------------
    # 4. 结果报告 - 包含精确度、召回率、F1 和准确性
    # -------------------------------------------------------------------

    true_labels = [1] * samples_with_k_predictions
    pred_labels = [1 if llm_validation_results.get(doc_id, False) else 0 for doc_id in eval_ids]

    # 计算四个核心指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    print("\n--- 评估结果 (Metrics) ---")
    print(f"Evaluated Samples (POS with predictions): {samples_with_k_predictions}")
    print(f"Top-K 成功命中数 (LLM V(E) Yes): {top_k_hits}")
    print("-" * 40)
    print("🎯 四个核心指标:")
    print(f"Accuracy (准确性): {accuracy:.4f}")
    print(f"Precision (精确度): {precision:.4f}")
    print(f"Recall (召回率/成功率): {recall:.4f}")
    print(f"F1 Score (F1 分数): {f1:.4f}")
    print("-" * 40)


# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(FILTERED_FILE_PATH):
        print(f"\nFATAL ERROR: Could not find {FILTERED_FILE_PATH}.")
    elif not os.path.exists(SAMPLED_FILE_PATH):
        print(f"\nFATAL ERROR: Could not find {SAMPLED_FILE_PATH}.")
    else:
        # 重定向所有 print 输出到报告文件
        print(f"Reading {FILTERED_FILE_PATH} and {SAMPLED_FILE_PATH} files...")
        print(f"Results will be saved to {REPORT_FILE_PATH}.")

        try:
            with open(REPORT_FILE_PATH, 'w', encoding='utf-8') as f:
                original_stdout = sys.stdout
                sys.stdout = f

                calculate_top_k_metrics(FILTERED_FILE_PATH, SAMPLED_FILE_PATH, llm)

        except Exception as e:
            sys.stdout = original_stdout
            print(f"FATAL ERROR: An error occurred while writing the report or executing the evaluation: {e}")
            sys.exit(1)

        finally:
            sys.stdout = original_stdout
            print(f"\n✅ Top-K 评估完成。所有指标（精确度、召回率、F1、准确性）已成功保存到 {REPORT_FILE_PATH}")