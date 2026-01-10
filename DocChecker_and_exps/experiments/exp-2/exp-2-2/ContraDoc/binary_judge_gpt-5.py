# -*- coding: utf-8 -*-
from knowledge_graph_maker import OpenAIClient
import json
import re
from dotenv import load_dotenv
import os
import random
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 初始化和配置 ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 建议使用 GPT-5 时保持 temperature 为 0 以获得最高稳定性
llm = OpenAIClient(
    model="gpt-5",
    temperature=0,
    top_p=0.9,
    max_tokens=1500
)

# --- 2. 增强型核心检测函数 ---

def detect_conflict_with_gpt(llm_client: Any, document_text: str, system_prompt: str) -> Tuple[int, str]:
    """
    使用正则表达式和更强的容错逻辑提取模型结果。
    """
    # 强制要求 JSON，并明确禁止过度推理
    system_prompt_json = (
        f"{system_prompt}\n\n"
        "### OUTPUT FORMAT ###\n"
        "You must output ONLY a valid JSON object. Do not include any preamble or postscript.\n"
        "JSON structure: {\"is_contradictory\": true/false, \"explanation\": \"string\"}"
    )

    default_explanation = "解析失败或模型未响应"

    try:
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt_json)
        if not response_text:
            return 0, default_explanation

        # 使用正则精准匹配最外层花括号
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # 注意：不要对整个 JSON 字符串做 lower()，否则解释中的语义会受损
            data = json.loads(json_str)

            # 标签解析
            val = data.get("is_contradictory", False)
            if isinstance(val, str):
                label = 1 if val.lower() == "true" else 0
            else:
                label = 1 if val is True else 0

            explanation = data.get("explanation", "No explanation provided.")
            return label, explanation
        else:
            return 0, "未能从响应中提取 JSON 块"

    except Exception as e:
        return 0, f"Error: {str(e)}"

# --- 3. 数据加载 (保持一致) ---

def load_all_contradoc_data(full_data: Dict) -> List[Dict[str, Any]]:
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})
    all_data_list = []
    for uid, info in pos_data.items():
        all_data_list.append({"unique_id": uid, "text": info["text"], "label": 1})
    for uid, info in neg_data.items():
        all_data_list.append({"unique_id": uid, "text": info["text"], "label": 0})
    random.shuffle(all_data_list)
    return all_data_list

# --- 4. 优化后的 SYSTEM PROMPT (Few-shot) ---

# 通过提供 Example 强制模型对齐“事实矛盾”的阈值
FEW_SHOT_PROMPT = (
    "You are an expert in logical consistency analysis. Your task is to detect direct factual contradictions within a document.\n"
    "A contradiction occurs when two parts of the text state facts that cannot both be true simultaneously.\n\n"
    "### CRITICAL GUIDELINES ###\n"
    "1. Focus on surface-level factual clashes (e.g., different dates, conflicting names, opposite outcomes).\n"
    "2. DO NOT try to rationalize or harmonize contradictions with creative assumptions.\n"
    "3. If one sentence says 'A' and another says 'Not A', it is TRUE (contradictory).\n\n"
    "### EXAMPLES ###\n"
    "Document: 'The meeting is at 2 PM. We will meet at 4 PM.'\n"
    "Output: {\"is_contradictory\": true, \"explanation\": \"The text lists two different times for the same meeting.\"}\n\n"
    "Document: 'The apple is red. It is a fruit.'\n"
    "Output: {\"is_contradictory\": false, \"explanation\": \"No conflicting facts detected.\"}\n"
)

# --- 5. 评估与保存逻辑 (结构同前，目录更新) ---

def save_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    SAVE_DIR = f"report/binary_judge_report_gpt-5"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 保存详细 JSON
    with open(os.path.join(SAVE_DIR, "details.json"), 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)

    # 保存指标 TXT
    with open(os.path.join(SAVE_DIR, "metrics.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n" + "-"*30 + "\n")
        for k, v in metrics.items():
            if k != "classification_report": f.write(f"{k}: {v}\n")
        f.write("\nClassification Report:\n" + metrics["classification_report"])

def evaluate_binary_detection_multithreaded(dataset: List[Dict[str, Any]], llm_client: Any, system_prompt: str, max_workers: int = 10):
    true_labels, pred_labels, detailed_results = [], [], []
    model_name = getattr(llm_client, 'model', 'gpt-5')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(detect_conflict_with_gpt, llm_client, item["text"], system_prompt): i for i, item in enumerate(dataset)}

        results = [None] * len(dataset)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            label, expl = future.result()
            results[idx] = (dataset[idx]["label"], label, expl, dataset[idx]["unique_id"])
            print(f"[{model_name}] Processed {sum(1 for x in results if x is not None)}/{len(dataset)}...")

    for t_label, p_label, expl, uid in results:
        true_labels.append(t_label)
        pred_labels.append(p_label)
        detailed_results.append({"unique_id": uid, "true": t_label, "pred": p_label, "explanation": expl})

    metrics = {
        "Accuracy": f"{accuracy_score(true_labels, pred_labels):.4f}",
        "F1_Score": f"{f1_score(true_labels, pred_labels, zero_division=0):.4f}",
        "Precision": f"{precision_score(true_labels, pred_labels, zero_division=0):.4f}",
        "Recall": f"{recall_score(true_labels, pred_labels, zero_division=0):.4f}",
        "classification_report": classification_report(true_labels, pred_labels, zero_division=0)
    }

    print("\n" + metrics["classification_report"])
    save_results_to_file(metrics, detailed_results, model_name)

if __name__ == "__main__":
    # 请确保路径正确
    DATA_PATH = "../../../../ContraDOC/sampled_contradoc.json"
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            dataset = load_all_contradoc_data(data)
            evaluate_binary_detection_multithreaded(dataset, llm, FEW_SHOT_PROMPT, max_workers=10)
    except Exception as e:
        print(f"运行失败: {e}")