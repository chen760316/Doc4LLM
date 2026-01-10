# -*- coding: utf-8 -*-
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import random
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import collections
import difflib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 配置和 LLM 客户端 ---
# 保持 LLM 配置不变
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

try:
    # 假设 gpt-4.1-mini 是一个有效的模型名称，如果不是，请替换为 gpt-4o 或 gpt-4
    llm = OpenAIClient(
        model="gpt-4o",
        temperature=0.1,
        top_p=0.9,
        max_tokens=1500
    )
except Exception as e:
    print(f"Error initializing OpenAIClient: {e}. Exiting.")
    exit()

# 定义结果保存目录
SAVE_DIR = "report/top_k_judge_report_gpt-4o"


# --- 2. 核心检测函数 (Top-K 证据定位) (保持不变) ---

def detect_top_k_evidence_with_gpt(llm_client: Any, document_text: str, system_prompt: str, k: int = 3) -> Dict[str, Any]:
    """
    使用 GPT 模型检测文档内冲突，返回 Top-K 证据和解释。
    """
    system_prompt_json = (
        f"{system_prompt}. "
        f"You must identify the {k} most likely text spans that, if true, would cause the document to be self-contradictory. "
        "Your output MUST be a single JSON object: "
        "{ \"top_k_evidence\": [\"span 1\", \"span 2\", ...], \"explanation\": \"[Your reason for selection.]\" }. "
        "Each span in 'top_k_evidence' must be an exact short quote from the document text. Output only the JSON."
    )

    default_result = {"top_k_evidence": [], "explanation": "无法获取模型响应或 JSON 解析失败。"}

    try:
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt_json)
    except Exception:
        return default_result

    try:
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text)

        evidence_list = response_json.get("top_k_evidence", [])
        explanation = response_json.get("explanation", default_result["explanation"]).strip().strip('"').strip("'")

        if not isinstance(evidence_list, list) or not all(isinstance(e, str) for e in evidence_list):
            evidence_list = []

        return {"top_k_evidence": evidence_list[:k], "explanation": explanation}

    except Exception:
        return default_result


# --- 3. 数据集加载函数 (加载所有 Positives, 取代采样函数) ---

def load_all_contradoc_data_pos(full_data: Dict) -> List[Dict[str, Any]]:
    """
    仅从 ContraDoc-POS 结构中加载所有正例数据，不进行任何抽样。
    """
    pos_data = full_data.get("pos", {})
    pos_list = []

    for unique_id, doc_info in pos_data.items():
        item = {
            "unique_id": unique_id,
            "text": doc_info["text"],
            "true_evidence": doc_info.get("evidence", ""),
            "type": doc_info.get("contra_type", ["Unknown"])[0] # 仅取第一个类型
        }
        pos_list.append(item)

    random.shuffle(pos_list)

    print(f"\n--- 数据加载统计 (仅 Positives) ---")
    print(f"正例 (矛盾) 总加载数: {len(pos_list)}")

    return pos_list


# --- 4. 评估主函数 (Top-K 评估 - 多线程版本) (保持不变) ---

def evaluate_top_k_detection_multithreaded(dataset: List[Dict[str, Any]], llm_client: Any, system_prompt: str, k: int = 3, max_workers: int = 10):
    """
    对数据集执行 Top-K 矛盾证据检测并计算成功率，并以伪二元分类指标形式报告结果。
    """
    total_samples = len(dataset)
    model_name = getattr(llm_client, 'model', 'MockLLM')
    print(f"\n--- 开始 Top-{k} 证据检测评估 ({total_samples} 个文档, 模型: {model_name}, 使用 {max_workers} 个线程) ---")

    future_to_index = {}
    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(dataset):
            doc_text = item["text"]
            future = executor.submit(detect_top_k_evidence_with_gpt, llm_client, doc_text, system_prompt, k)
            future_to_index[future] = i

        results_list = [None] * total_samples

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            item = dataset[i]
            unique_id = item["unique_id"]
            true_evidence = item["true_evidence"].strip()

            try:
                model_output = future.result()
            except Exception as exc:
                print(f"线程执行任务 {unique_id} 时发生异常: {exc}")
                model_output = {"top_k_evidence": [], "explanation": f"执行异常: {exc}"}

            results_list[i] = (true_evidence, model_output, unique_id)
            print(f"✅ 已处理 {i + 1}/{total_samples} 个文档...")

    end_time = datetime.now()
    print(f"\n评估完成！总耗时: {(end_time - start_time).total_seconds():.2f} 秒.")

    # --- 结果汇总和指标计算 ---

    successful_detections = 0
    detailed_results = []

    # 伪二元分类指标
    true_labels = [1] * total_samples  # 仅 POS 样本，真实标签始终为 1 (Hit)
    pred_labels = []

    for i, (true_evidence, model_output, unique_id) in enumerate(results_list):
        predicted_evidence = model_output["top_k_evidence"]
        explanation = model_output["explanation"]
        is_hit = False

        # 检查真实证据是否在 Top-K 预测证据中 (模糊匹配)
        for pred_span in predicted_evidence:
            if difflib.SequenceMatcher(None, true_evidence.lower(), pred_span.lower()).ratio() >= 0.8 or \
                    true_evidence.lower() in pred_span.lower() or \
                    pred_span.lower() in true_evidence.lower():
                is_hit = True
                break

        if is_hit:
            successful_detections += 1
            pred_labels.append(1)
        else:
            pred_labels.append(0)

        # 收集详细结果
        result_entry = {
            "unique_id": unique_id,
            "true_evidence": true_evidence,
            "predicted_top_k": predicted_evidence,
            "is_hit_in_top_k": is_hit,
            "explanation": explanation
        }
        detailed_results.append(result_entry)

        # 打印命中信息
        status_symbol = "✅ HIT" if is_hit else "❌ MISS"
        print(f"[{status_symbol}] ID: {unique_id} | 命中: {is_hit}")

    # 计算最终指标
    success_rate = successful_detections / total_samples if total_samples > 0 else 0

    # 伪二元分类指标计算
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    # 分类报告（仅 1 类）
    cls_report = classification_report(true_labels, pred_labels, target_names=["Miss (0)", "Hit (1)"], zero_division=0)


    print("\n--- 评估结果 (Metrics) ---")
    print(f"Top-{k} 证据检测成功数: {successful_detections} / {total_samples}")
    print(f"Top-{k} 成功率 (Success Rate / Recall / Accuracy): {success_rate:.4f}")
    print(f"Precision (精确度): {precision:.4f} (注意：在仅 POS 设置下，该值倾向于 1.0)")
    print(f"F1 Score (F1 分数): {f1:.4f}")
    print("\n--- 详细分类报告 ---")
    print(cls_report)


    # 准备保存的指标字典
    metrics_to_save = {
        "Model": model_name,
        "Total_Samples": total_samples,
        f"Top_{k}_Success_Rate": f"{success_rate:.4f}",
        "Hit_Count": successful_detections,
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1_Score": f"{f1:.4f}",
        "classification_report": cls_report
    }

    # 保存结果到文件
    save_top_k_results_to_file(metrics_to_save, detailed_results, model_name, k)


# --- 5. 结果保存辅助函数 (已修改保存路径) (保持不变) ---

def save_top_k_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str, k: int):
    """保存 Top-K 评估指标和详细的文档结果到本地文件，保存至 report/top_k_judge_report 目录下。"""

    # 创建目录（如果不存在）
    os.makedirs(SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ContraDoc_Top{k}_Eval_{model_name}_{timestamp}"

    # 1. 保存详细结果 (JSON)
    json_path = os.path.join(SAVE_DIR, f"{base_filename}_details.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"\n✨ 详细评估结果已保存到: {json_path}")

    # 2. 保存指标结果 (TXT)
    txt_path = os.path.join(SAVE_DIR, f"{base_filename}_metrics.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ContraDoc Top-{k} 证据检测评估指标 ({model_name}) ---\n")
        f.write(f"时间: {timestamp}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Top-{k} 成功率 (Success Rate): {metrics.get(f'Top_{k}_Success_Rate', 'N/A')}\n")
        f.write("-" * 40 + "\n")
        f.write("伪二元分类指标 (Top-K Hit vs Miss):\n")
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

    TOP_K = 5
    # 删除了 SAMPLES 变量，因为我们加载所有数据
    MAX_CONCURRENT_CALLS = 10

    try:
        # 假设文件路径在您的环境中是正确的
        # ⚠️ 请确保路径 "../../../../ContraDOC/sampled_contradoc.json" 确实指向包含 "pos" 键的 ContraDoc 数据集
        data_path = "../../../../ContraDOC/sampled_contradoc.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print("Error: ContraDoc.json not found. Please check the file path.")
        exit()
    except json.JSONDecodeError:
        print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
        exit()

    # *** 核心修改：加载所有正例数据 (不采样) ***
    contradoc_dataset = load_all_contradoc_data_pos(full_data)

    # 定义系统提示 (用于 Top-K 定位)
    SYSTEM_PROMPT = (
        "You are an expert natural language processing model specialized in identifying the specific spans of text "
        "that cause self-contradiction within a single document. "
        "Your task is to analyze the document and return the most crucial text spans that contribute to the contradiction."
    )

    if len(contradoc_dataset) > 0:
        # 执行 Top-K 评估 (使用多线程版本)
        evaluate_top_k_detection_multithreaded(
            contradoc_dataset,
            llm,
            SYSTEM_PROMPT,
            k=TOP_K,
            max_workers=MAX_CONCURRENT_CALLS
        )
    else:
        print("警告: 数据集中没有找到正例 (矛盾) 文档，评估未执行。")