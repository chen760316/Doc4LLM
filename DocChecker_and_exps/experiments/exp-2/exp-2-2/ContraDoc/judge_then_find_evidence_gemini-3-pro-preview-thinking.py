# -*- coding: utf-8 -*-
from openai import OpenAI, APIError  # 引入 OpenAI 和 APIError
import json
from dotenv import load_dotenv
import os
import random
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, precision_recall_fscore_support
import collections
import difflib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import sys
import time

# --- 1. 配置和 LLM 客户端 (替换为 Gemini 配置) ---
print("--- 1. 正在加载配置和初始化 Gemini Client (OpenAI 兼容模式) ---")
load_dotenv()

# Gemini 配置 (使用您的代码片段中的变量名)
API_KEY = os.getenv("GEMINI_API_KEY")
# 使用您测试代码中的模型名称
GEMINI_MODEL_NAME = "gemini-3-pro-preview-thinking"
MODEL_NAME = GEMINI_MODEL_NAME


print(f"✅ 使用模型: {MODEL_NAME}")

# LLM 客户端初始化，使用 OpenAI 兼容客户端
try:
    # 关键：使用 Gemini 的 Key 和中转站 Base URL 初始化 OpenAI Client
    client = OpenAI(
        api_key=API_KEY
    )
    print(f"✅ Gemini Client (通过 OpenAI 兼容接口) 初始化成功")
except Exception as e:
    print(f"❌ 错误: Client 初始化失败: {e}")
    sys.exit(1)

# 定义 BERTScore 阈值
BERT_SCORE_THRESHOLD = 0.98
# 定义结果保存目录 (更新为 Gemini)
SAVE_DIR = "report/judge_then_find_evidence_report_gemini-3-pro-preview-thinking"


# --- 辅助函数：格式化 LaTeX 表格行 (更新模型名处理) ---
def format_latex_table_row(model_name: str, jd_f1: float, jdv_f1: float, ehr: float, strict_p: float, strict_r: float) -> str:
    """
    Formats the evaluation results into a LaTeX table row, retaining four decimal places.
    """
    # 格式化模型名称：针对 Gemini 2.5 Pro
    if "gemini-2.5-pro" in model_name.lower():
        display_model_name = "\\Gemini-2.5-Pro"
    else:
        # 其他模型使用默认格式
        display_model_name = model_name.replace('-', ' ').title().replace('Gpt', 'GPT').replace('O', 'o').replace('Mini', 'Mini')

    # 转换为小数点后四位的字符串
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


# --- 2. BERTScore 语义匹配模拟函数 (保持不变) ---

def semantic_match_bertscore_logic(predicted_evidence: List[str], true_evidence: str, threshold: float) -> bool:
    """
    模拟使用 BERTScore 逻辑来判断证据是否正确。
    """
    if not predicted_evidence or not true_evidence:
        return False

    true_evidence_lower = true_evidence.strip().lower()

    for pred_span in predicted_evidence:
        pred_span_lower = pred_span.strip().lower()

        # 严格的模糊匹配 (模拟高分 BERTScore)
        if difflib.SequenceMatcher(None, true_evidence_lower, pred_span_lower).ratio() >= 0.95:
            # 模拟 BERTScore > 0.98 的匹配成功
            return True

        # 即使模糊匹配不成功，如果模型证据完全包含真实证据，或反之，也视为成功。
        elif true_evidence_lower in pred_span_lower and len(true_evidence_lower) > 5:
            return True
        elif pred_span_lower in true_evidence_lower and len(pred_span_lower) > 5:
            return True

    return False


# --- 3. 核心检测函数 (判断 + 证据) (适配标准的 OpenAI 客户端) ---

def detect_combined_task_with_gpt(llm_client: OpenAI, document_text: str, system_prompt: str) -> Dict[str, Any]:
    """
    使用标准的 OpenAI Client (适配 Gemini/中转站) 执行判断+证据提取任务，并加入重试机制。
    """
    system_prompt_json = (
        f"{system_prompt}. Your output MUST be a single JSON object: "
        "{ \"is_contradictory\": true/false, \"supporting_evidence\": [\"[Quote Sentence 1]\", \"[Quote Sentence 2]\"], \"explanation\": \"[Your reason or evidence for the decision.]\" }. "
        "The 'supporting_evidence' array must be empty if 'is_contradictory' is false. Do not include any other text outside the JSON block. Output only the JSON."
    )

    default_result = {"label": 0, "evidence": [], "explanation": "无法获取模型响应或 JSON 解析失败。"}
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    response_text = ""  # 初始化 response_text

    for attempt in range(MAX_RETRIES):
        try:
            # --- 替换为标准的 OpenAI chat.completions.create 方法 ---
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,  # 使用全局 Gemini 模型名
                messages=[
                    {"role": "system", "content": system_prompt_json},
                    {"role": "user", "content": document_text}
                ],
                temperature=0.1,
                max_tokens=1500,
                timeout=30  # 增加超时设置
            )

            if not response.choices or not response.choices[0].message:
                raise Exception("Empty response from API.")

            response_text = response.choices[0].message.content
            # --- API 调用成功，跳出重试循环 ---
            break
        except APIError as e:
            # 503 Service Unavailable, 429 Rate Limit, 或其他瞬时错误
            if e.status_code in [500, 502, 503, 504, 429] and attempt < MAX_RETRIES - 1:
                print(f"[⚠️ RETRY] API Error: {e.status_code}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                # 致命错误 (如 400 Bad Request, 404 Not Found) 或达到最大重试次数
                print(f"API Error during call: {e}")
                return default_result
        except Exception:
            # 其他网络或连接错误
            if attempt < MAX_RETRIES - 1:
                print(f"[⚠️ RETRY] General Error. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                return default_result
    else:
        # 如果循环没有通过 break 退出 (即达到最大重试次数)
        return default_result

    try:
        # --- 增强 JSON 提取逻辑 ---
        start = response_text.find("```json")
        end = response_text.rfind("```")

        if start != -1 and end > start:
            json_str = response_text[start + len("```json"):end].strip()
        else:
            json_str = response_text.strip()

        if not json_str:
            raise ValueError("Extracted JSON string is empty.")

        response_json = json.loads(json_str)

        is_contradictory = str(response_json.get("is_contradictory", False)).lower()
        label = 1 if is_contradictory == "true" else 0
        evidence_list = response_json.get("supporting_evidence", [])
        if not isinstance(evidence_list, list): evidence_list = []
        explanation = response_json.get("explanation", "N/A").strip().strip('"').strip("'")

        return {"label": label, "evidence": evidence_list, "explanation": explanation}

    except Exception as e:
        # 打印调试信息，帮助用户理解为何 JSON 解析失败
        print(f"\n[❌ JSON PARSE ERROR] ID: N/A | Error: {e}")
        # 检查 response_text 是否为空，或者打印前200字符
        debug_output = response_text.strip()
        if not debug_output:
            print("[DEBUG] Original Response Text was empty or non-string.")
        else:
            print(f"[DEBUG] Original Response Text (First 200 chars): {debug_output[:200]}")

        return default_result


# --- 4. 数据集加载和抽样函数 (保持不变) ---

def load_and_sample_contradoc_data_combined(full_data: Dict, num_pos: int = 40, num_neg: int = 40) -> List[Dict[str, Any]]:
    # ... (保持不变)
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    pos_list = []
    pos_grouped = collections.defaultdict(list)
    for unique_id, doc_info in pos_data.items():
        types = doc_info.get("contra_type", ["Unknown"])
        for c_type in types:
            pos_grouped[c_type].append({
                "unique_id": unique_id,
                "text": doc_info["text"],
                "label": 1,
                "true_evidence": doc_info.get("evidence", ""),
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

    neg_available = []
    for unique_id, doc_info in neg_data.items():
        neg_available.append({
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 0,
            "true_evidence": ""
        })

    num_to_sample_neg = min(num_neg, len(neg_available))
    neg_list = random.sample(neg_available, num_to_sample_neg)

    print(f"\n--- 数据抽样统计 ---")
    print(f"正例 (矛盾) 抽取数: {len(pos_list)} / 目标 {num_pos}")
    print(f"负例 (非矛盾) 抽取数: {len(neg_list)} / 目标 {num_neg}")

    dataset = pos_list + neg_list
    random.shuffle(dataset)
    return dataset


# --- 5. 评估主函数 (多线程 + J(D)∧V(E) 严格正确性) (保持不变) ---

def evaluate_combined_detection_multithreaded(dataset: List[Dict[str, Any]], llm_client: Any, system_prompt: str, max_workers: int = 10):
    # ... (保持不变)
    future_to_index = {}
    model_name = MODEL_NAME  # 使用全局模型名
    print(f"\n--- 开始组合任务评估 ({len(dataset)} 个文档, 模型: {model_name}, 使用 {max_workers} 个线程) ---")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(dataset):
            doc_text = item["text"]
            # 提交任务，使用全局模型客户端
            future = executor.submit(detect_combined_task_with_gpt, llm_client, doc_text, system_prompt)
            future_to_index[future] = i

        results_list = [None] * len(dataset)
        processed_count = 0

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            item = dataset[i]
            unique_id = item["unique_id"]

            try:
                model_output = future.result()
            except Exception as exc:
                print(f"线程执行任务 {unique_id} 时发生异常: {exc}")
                model_output = {"label": 0, "evidence": [], "explanation": f"执行异常: {exc}"}

            results_list[i] = (item["label"], item["true_evidence"].strip(), model_output, unique_id)
            processed_count += 1
            print(f"✅ 已处理 {processed_count}/{len(dataset)} 个文档...")

    # --- 结果汇总和指标计算 (保持不变) ---

    # 严格正确性指标 (J(D) ∧ V(E))
    strict_correct_labels = []
    strict_predicted_labels = []

    # 二元判断指标 (J(D))
    binary_true_labels = []
    binary_pred_labels = []

    detailed_results = []
    pos_sample_count = 0
    predicted_pos_count = 0
    evidence_hit_on_pos_and_yes = 0

    for true_label, true_evidence, model_output, unique_id in results_list:

        predicted_label = model_output["label"]
        predicted_evidence = model_output["evidence"]
        explanation = model_output["explanation"]

        # --- J(D) 二元判断指标收集 ---
        binary_true_labels.append(true_label)
        binary_pred_labels.append(predicted_label)

        # --- 证据评估逻辑 (V(E)) ---
        is_evidence_correct = False

        if true_label == 1:
            pos_sample_count += 1

            # V(E): 只有当真实标签为 1 时才评估证据
            is_evidence_correct = semantic_match_bertscore_logic(
                predicted_evidence, true_evidence, BERT_SCORE_THRESHOLD
            )

            if predicted_label == 1:
                predicted_pos_count += 1  # 只有真实矛盾中，模型预测为“是”，才计入 EHR 分母

                if is_evidence_correct:
                    evidence_hit_on_pos_and_yes += 1

        # --- J(D) ∧ V(E) 严格正确性指标收集 ---

        # 1. 严格正确性：真实标签
        strict_correct_labels.append(true_label)

        # 2. 严格正确性：预测标签
        # 负样本 (true_label == 0)：只有模型预测 $J(D)=0$ 才是严格正确。
        if true_label == 0:
            strict_predicted_result = 1 if predicted_label == 0 else 0
        else:  # true_label == 1:
            # 正样本：只有 $J(D)_{pred}=1$ 且 $V(E)=1$ 才是严格正确。
            strict_predicted_result = 1 if predicted_label == 1 and is_evidence_correct else 0

        strict_predicted_labels.append(strict_predicted_result)

        # 收集详细结果
        result_entry = {
            "unique_id": unique_id,
            "true_label": true_label,
            "predicted_binary_label": predicted_label,
            "true_evidence": true_evidence if true_label == 1 else "N/A",
            "predicted_evidence": predicted_evidence,
            "V(E)_Correct": is_evidence_correct if true_label == 1 else "N/A",
            "J(D)∧V(E)_Strict_Correctness": strict_predicted_result == true_label,
            "explanation": explanation,
        }
        detailed_results.append(result_entry)

        # 打印证据命中信息
        hit_symbol = "✅ HIT" if (true_label == 1 and is_evidence_correct) else "❌ MISS"
        if true_label == 1:
            print(f"[{hit_symbol}] ID: {unique_id} | J(D)_pred: {predicted_label} | V(E): {is_evidence_correct} | Strict Correctness: {strict_predicted_result == true_label}")

    # --- 最终指标计算 ---

    # 1. J(D) ∧ V(E) 严格正确性指标
    strict_accuracy = accuracy_score(strict_correct_labels, strict_predicted_labels)
    (strict_precision, strict_recall, strict_f1, _) = precision_recall_fscore_support(
        strict_correct_labels, strict_predicted_labels, average='binary', pos_label=1, zero_division=0
    )
    strict_cls_report = classification_report(strict_correct_labels, strict_predicted_labels, target_names=["Not Correct (0)", "Strictly Correct (1)"], zero_division=0)

    # 2. J(D) 二元判断指标 (重新计算并确保报告)
    binary_accuracy = accuracy_score(binary_true_labels, binary_pred_labels)
    binary_precision = precision_score(binary_true_labels, binary_pred_labels, zero_division=0)
    binary_recall = recall_score(binary_true_labels, binary_pred_labels, zero_division=0)
    binary_f1 = f1_score(binary_true_labels, binary_pred_labels, zero_division=0)

    # 3. EHR (Evidence Hit Rate)
    ehr = evidence_hit_on_pos_and_yes / predicted_pos_count if predicted_pos_count > 0 else 0

    print("\n--- 评估结果 (Metrics) ---")
    print("--- 1. J(D) ∧ V(E) 严格正确性指标 ---")
    print(f"Strict Accuracy (严格准确性): {strict_accuracy:.4f}")
    print(f"Strict Precision (严格精确度): {strict_precision:.4f}")
    print(f"Strict Recall (严格召回率): {strict_recall:.4f}")
    print(f"Strict F1 Score (严格 F1 分数): {strict_f1:.4f}")

    print("\n--- 2. J(D) 二元判断指标 (仅供参照) ---")
    print(f"Binary Accuracy (二元准确性): {binary_accuracy:.4f}")
    print(f"Binary Precision (二元精确度): {binary_precision:.4f}")
    print(f"Binary Recall (二元召回率): {binary_recall:.4f}")
    print(f"Binary F1 Score (二元 F1 分数): {binary_f1:.4f}")

    print("\n--- 3. EHR (证据命中率) ---")
    print(f"模型预测 '是' (J(D)=1) 的 POS 样本数: {predicted_pos_count}")
    print(f"V(E) 成功命中数: {evidence_hit_on_pos_and_yes}")
    print(f"EHR (Evidence Hit Rate): {ehr:.4f}")

    print("\n--- 详细分类报告 (J(D) ∧ V(E)) ---")
    print(strict_cls_report)

    # 准备保存的指标字典
    metrics_to_save = {
        "Model": model_name,
        "Total_Samples": len(dataset),
        "POS_Samples": pos_sample_count,
        # J(D) ∧ V(E) 严格指标
        "Strict_Accuracy": f"{strict_accuracy:.4f}",
        "Strict_Precision": f"{strict_precision:.4f}",
        "Strict_Recall": f"{strict_recall:.4f}",
        "Strict_F1_Score": f"{strict_f1:.4f}",
        # J(D) 二元指标
        "Binary_Accuracy": f"{binary_accuracy:.4f}",
        "Binary_Precision": f"{binary_precision:.4f}",
        "Binary_Recall": f"{binary_recall:.4f}",
        "Binary_F1_Score": f"{binary_f1:.4f}",
        # EHR 指标
        "EHR": f"{ehr:.4f}",
        "V(E)_Hit_Count": evidence_hit_on_pos_and_yes,
        "Predicted_Pos_Count": predicted_pos_count,
        "Strict_classification_report": strict_cls_report
    }

    # --- 关键：输出 LaTeX 行 ---
    latex_row = format_latex_table_row(
        model_name=model_name,
        jd_f1=binary_f1,
        jdv_f1=strict_f1,
        ehr=ehr,
        strict_p=strict_precision,
        strict_r=strict_recall
    )

    print("\n" + "=" * 50)
    print("--- LaTeX Table Row Output (4 Decimal Places) ---")
    print(latex_row.strip())
    print("=" * 50 + "\n")

    # 保存结果到文件
    save_combined_results_to_file(metrics_to_save, detailed_results, model_name)


# --- 6. 结果保存辅助函数 (保持不变) ---

def save_combined_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存组合任务评估指标和详细的文档结果到本地文件，保存至 report/judge_then_find_evidence_report 目录下。"""

    # 定义目标目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ContraDoc_Strict_Eval_{model_name}_{timestamp}"

    # 定义 LaTeX 文件路径
    LATEX_TABLE_FILE_PATH = os.path.join(SAVE_DIR, f"{base_filename}_latex_row.txt")

    # 1. 保存详细结果 (JSON)
    json_path = os.path.join(SAVE_DIR, f"{base_filename}_details.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"\n✨ 详细评估结果已保存到: {json_path}")

    # 2. 保存指标结果 (TXT)
    txt_path = os.path.join(SAVE_DIR, f"{base_filename}_metrics.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ContraDoc 严格组合任务评估指标 ({model_name}) ---\n")
        f.write(f"时间: {timestamp}\n")
        f.write("-" * 40 + "\n")
        f.write("1. J(D) ∧ V(E) 严格正确性指标:\n")
        f.write(f"Strict Accuracy (严格准确性): {metrics.get('Strict_Accuracy', 'N/A')}\n")
        f.write(f"Strict Precision (严格精确度): {metrics.get('Strict_Precision', 'N/A')}\n")
        f.write(f"Strict Recall (严格召回率): {metrics.get('Strict_Recall', 'N/A')}\n")
        f.write(f"Strict F1 Score (严格 F1 分数): {metrics.get('Strict_F1_Score', 'N/A')}\n")
        f.write("-" * 40 + "\n")
        f.write("2. EHR (Evidence Hit Rate):\n")
        f.write(f"EHR: {metrics.get('EHR', 'N/A')}\n")
        f.write(f"V(E) 命中数 / 模型预测 '是' 数: {metrics.get('V(E)_Hit_Count', 'N/A')} / {metrics.get('Predicted_Pos_Count', 'N/A')}\n")
        f.write("-" * 40 + "\n")
        f.write("3. J(D) 二元判断指标 (仅供参照):\n")
        f.write(f"Binary Accuracy: {metrics.get('Binary_Accuracy', 'N/A')}\n")
        f.write(f"Binary Precision: {metrics.get('Binary_Precision', 'N/A')}\n")
        f.write(f"Binary Recall: {metrics.get('Binary_Recall', 'N/A')}\n")
        f.write(f"Binary F1 Score: {metrics.get('Binary_F1_Score', 'N/A')}\n")
        f.write("-" * 40 + "\n")
        f.write("详细分类报告 (J(D) ∧ V(E)):\n")
        f.write(metrics.get("Strict_classification_report", "N/A"))

    print(f"✨ 最终评估指标已保存到: {txt_path}")

    # 3. 保存 LaTeX 行 (单独文件)
    latex_row = format_latex_table_row(
        model_name=model_name,
        jd_f1=float(metrics["Binary_F1_Score"]),
        jdv_f1=float(metrics["Strict_F1_Score"]),
        ehr=float(metrics["EHR"]),
        strict_p=float(metrics["Strict_Precision"]),
        strict_r=float(metrics["Strict_Recall"])
    )

    with open(LATEX_TABLE_FILE_PATH, 'w', encoding='utf-8') as f_latex:
        f_latex.write(latex_row)
    print(f"✨ LaTeX table row saved to: {LATEX_TABLE_FILE_PATH}")


# --- 7. 执行脚本 ---

if __name__ == "__main__":

    SAMPLES_PER_CLASS = 40
    MAX_CONCURRENT_CALLS = 10

    try:
        # 简化路径查找，只保留一个通用路径
        data_path = "../../../../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: ContraDoc.json not found. Please check paths: {data_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
        sys.exit(1)

    contradoc_dataset = load_and_sample_contradoc_data_combined(full_data, num_pos=SAMPLES_PER_CLASS, num_neg=SAMPLES_PER_CLASS)

    SYSTEM_PROMPT = (
        "Given a document, the model needs to determine whether there is self-contradiction by answering 'true' or 'false'. "
        "If the answer is 'true', the model must also provide supporting evidence by quoting the sentence(s) "
        "that indicate the self-contradiction within the document. "
        "The 'supporting_evidence' field must contain the quoted text spans."
    )

    evaluate_combined_detection_multithreaded(
        contradoc_dataset,
        client,  # 使用全局创建的 Gemini 客户端
        SYSTEM_PROMPT,
        max_workers=MAX_CONCURRENT_CALLS
    )