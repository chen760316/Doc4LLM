"""
使用的数据集：ContraDoc
使用的基准："gpt-4.1-mini"
二元判断的另一个缺点是，回答“是”并不一定意味着该模型可以找到自相矛盾之处。
我们设计了另一个任务，当回答“是”时，不仅要给出二元判断，还要给出自相矛盾的证据句。在此设置中，模型在CONTRADOC上进行评估
"""
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import random
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import collections
import difflib  # 用于证据匹配
from datetime import datetime

# --- 1. 配置和 LLM 客户端 ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

try:
    llm = OpenAIClient(
        model="gpt-4.1-mini",
        temperature=0.1,
        top_p=0.9,
        max_tokens=1500
    )
except Exception as e:
    print(f"Error initializing OpenAIClient: {e}. Exiting.")
    exit()


# --- 2. 核心检测函数 (判断 + 证据) ---

def detect_combined_task_with_gpt(llm_client: Any, document_text: str, system_prompt: str) -> Dict[str, Any]:
    """
    使用 GPT 模型执行二元判断和证据查找的组合任务。
    返回: {"label": int, "evidence": List[str], "explanation": str}
    """
    # 要求 JSON 包含二元判断和证据列表
    system_prompt_json = (
        f"{system_prompt}. "
        "Your output MUST be a single JSON object: "
        "{ \"is_contradictory\": true/false, \"supporting_evidence\": [\"evidence span 1\", \"span 2\", ...], \"explanation\": \"[Your reason.]\" }. "
        "If 'is_contradictory' is false, 'supporting_evidence' should be an empty list []."
        "The evidence spans must be quotes from the document."
    )

    default_result = {"label": 0, "evidence": [], "explanation": "无法获取模型响应或 JSON 解析失败。"}

    try:
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt_json)
    except Exception:
        return default_result

    try:
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text)

        # 1. 提取标签
        is_contradictory = str(response_json.get("is_contradictory", False)).lower()
        label = 1 if is_contradictory == "true" else 0

        # 2. 提取证据
        evidence_list = response_json.get("supporting_evidence", [])
        if not isinstance(evidence_list, list):
            evidence_list = []  # 确保证据是列表

        # 3. 提取解释
        explanation = response_json.get("explanation", "N/A").strip().strip('"').strip("'")

        return {"label": label, "evidence": evidence_list, "explanation": explanation}

    except Exception:
        return default_result


# --- 3. 数据集加载和抽样函数 (恢复 POS 和 NEG) ---

def load_and_sample_contradoc_data_combined(full_data: Dict, num_pos: int = 40, num_neg: int = 40) -> List[Dict[str, Any]]:
    """
    从 ContraDoc JSON 结构中加载数据，并按照要求进行抽样，包含 POSitives 和 NEGatives。
    - POSitives 包含 'true_evidence'。
    """
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    # 使用 Top-K 任务中的 POS 抽样逻辑，以保持类型均衡
    # ----------------------------------------------------
    pos_list = []
    pos_grouped = collections.defaultdict(list)
    for unique_id, doc_info in pos_data.items():
        types = doc_info.get("contra_type", ["Unknown"])
        for c_type in types:
            pos_grouped[c_type].append({
                "unique_id": unique_id,
                "text": doc_info["text"],
                "label": 1,
                # 关键：POSitives 必须包含真实矛盾证据
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
    # ----------------------------------------------------

    # 负例 (Negatives) 随机抽样
    neg_available = []
    for unique_id, doc_info in neg_data.items():
        neg_available.append({
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 0,
            "true_evidence": ""  # 负例无证据
        })

    num_to_sample_neg = min(num_neg, len(neg_available))
    neg_list = random.sample(neg_available, num_to_sample_neg)

    print(f"\n--- 数据抽样统计 ---")
    print(f"正例 (矛盾) 抽取数: {len(pos_list)} / 目标 {num_pos}")
    print(f"负例 (非矛盾) 抽取数: {len(neg_list)} / 目标 {num_neg}")

    dataset = pos_list + neg_list
    random.shuffle(dataset)
    return dataset


# --- 4. 评估主函数 (组合任务评估) ---

def evaluate_combined_detection(dataset: List[Dict[str, Any]], llm_client: Any, system_prompt: str):
    """
    对数据集执行组合任务评估，计算二元指标和 POS 上的证据定位成功率。
    """
    true_labels = []
    pred_labels = []
    detailed_results = []

    pos_sample_count = 0
    evidence_hit_count = 0

    model_name = getattr(llm_client, 'model', 'MockLLM')
    print(f"\n--- 开始组合任务评估 ({len(dataset)} 个文档, 模型: {model_name}) ---")

    for i, item in enumerate(dataset):
        doc_text = item["text"]
        unique_id = item["unique_id"]
        true_label = item["label"]
        true_evidence = item["true_evidence"].strip()

        # 获取模型的输出
        model_output = detect_combined_task_with_gpt(llm_client, doc_text, system_prompt)
        predicted_label = model_output["label"]
        predicted_evidence = model_output["evidence"]
        explanation = model_output["explanation"]

        true_labels.append(true_label)
        pred_labels.append(predicted_label)

        is_evidence_hit = False

        if true_label == 1:  # 仅在 POSitives 上评估证据定位能力
            pos_sample_count += 1

            # 只有当模型预测为矛盾 (1) 且提供了证据时，才进行证据匹配
            if predicted_label == 1 and predicted_evidence:
                # 检查真实证据是否包含在模型预测的任何证据中
                for pred_span in predicted_evidence:
                    # 使用模糊匹配检查 (与 Top-K 任务相同逻辑)
                    if difflib.SequenceMatcher(None, true_evidence.lower(), pred_span.lower()).ratio() >= 0.8 or \
                            true_evidence.lower() in pred_span.lower() or \
                            pred_span.lower() in true_evidence.lower():
                        is_evidence_hit = True
                        break

                if is_evidence_hit:
                    evidence_hit_count += 1

            # 打印证据命中信息
            hit_symbol = "✅ HIT" if is_evidence_hit else "❌ MISS"
            if true_label == 1:
                print(f"[{hit_symbol}] ID: {unique_id} | Type: POS | 证据定位: {is_evidence_hit}")

        # 收集详细结果
        result_entry = {
            "unique_id": unique_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "true_evidence": true_evidence if true_label == 1 else "N/A",
            "predicted_evidence": predicted_evidence,
            "evidence_match_on_pos": is_evidence_hit if true_label == 1 else "N/A",
            "explanation": explanation,
            "is_correct_binary": (true_label == predicted_label)
        }
        detailed_results.append(result_entry)

        if (i + 1) % 10 == 0 or i == len(dataset) - 1:
            print(f"✅ 已处理 {i + 1}/{len(dataset)} 个文档...")

    # 1. 计算二元判断指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    cls_report = classification_report(true_labels, pred_labels, target_names=["Non-Contradictory (0)", "Contradictory (1)"], zero_division=0)

    # 2. 计算证据定位成功率
    evidence_success_rate = evidence_hit_count / pos_sample_count if pos_sample_count > 0 else 0

    print("\n--- 评估结果 (Metrics) ---")
    print("--- 1. 二元判断指标 ---")
    print(f"Accuracy (准确性): {accuracy:.4f}")
    print(f"Precision (精确度): {precision:.4f}")
    print(f"Recall (召回率): {recall:.4f}")
    print(f"F1 Score (F1 分数): {f1:.4f}")

    print("\n--- 2. 证据定位指标 (仅针对 POS 样本) ---")
    print(f"POS 样本总数: {pos_sample_count}")
    print(f"证据命中数: {evidence_hit_count}")
    print(f"证据定位成功率 (Evidence Success Rate): {evidence_success_rate:.4f}")
    print("\n--- 详细分类报告 (二元判断) ---")
    print(cls_report)

    # 准备保存的指标字典
    metrics_to_save = {
        "Model": model_name,
        "Total_Samples": len(dataset),
        "POS_Samples": pos_sample_count,
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1_Score": f"{f1:.4f}",
        "Evidence_Hit_Count": evidence_hit_count,
        "Evidence_Success_Rate": f"{evidence_success_rate:.4f}",
        "classification_report": cls_report
    }

    # 保存结果到文件
    save_combined_results_to_file(metrics_to_save, detailed_results, model_name)


# --- 5. 结果保存辅助函数 (为组合任务调整) ---

def save_combined_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存组合任务评估指标和详细的文档结果到本地文件。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ContraDoc_Combined_Eval_{model_name}_{timestamp}"

    # 1. 保存详细结果 (JSON)
    json_path = f"{base_filename}_details.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"\n✨ 详细评估结果已保存到: {json_path}")

    # 2. 保存指标结果 (TXT)
    txt_path = f"{base_filename}_metrics.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ContraDoc 组合任务评估指标 ({model_name}) ---\n")
        f.write(f"时间: {timestamp}\n")
        f.write("-" * 40 + "\n")
        f.write("二元判断指标:\n")
        f.write(f"Accuracy: {metrics['Accuracy']}\n")
        f.write(f"F1_Score: {metrics['F1_Score']}\n")
        f.write("\n证据定位指标 (仅POS):\n")
        f.write(f"POS_Samples: {metrics['POS_Samples']}\n")
        f.write(f"Evidence_Success_Rate: {metrics['Evidence_Success_Rate']}\n")
        f.write("-" * 40 + "\n")
        f.write("详细分类报告 (二元判断):\n")
        f.write(metrics.get("classification_report", "N/A"))

    print(f"✨ 最终评估指标已保存到: {txt_path}")


# --- 6. 执行脚本 ---

if __name__ == "__main__":

    SAMPLES_PER_CLASS = 40  # POS 和 NEG 各抽样 40 篇

    try:
        with open("../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json", 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print("Error: ContraDoc.json not found. Please check the file path.")
        exit()
    except json.JSONDecodeError:
        print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
        exit()

    # 抽样 POS 和 NEG
    contradoc_dataset = load_and_sample_contradoc_data_combined(full_data, num_pos=SAMPLES_PER_CLASS, num_neg=SAMPLES_PER_CLASS)

    # 定义系统提示 (用于组合任务)
    SYSTEM_PROMPT = (
        "Given a document, the model needs to determine whether there is self-contradiction by answering 'true' or 'false'. "
        "If the answer is 'true', the model must also provide supporting evidence by quoting the sentence(s) "
        "that indicate the self-contradiction within the document. "
        "The 'supporting_evidence' field must contain the quoted text spans."
    )

    # 执行组合任务评估
    evaluate_combined_detection(contradoc_dataset, llm, SYSTEM_PROMPT)