"""
使用的数据集：ContraDoc
使用的基准："gpt-4.1-mini"
在zero-shot设置下，模型在特定任务上的表现受其对自我矛盾的敏感性的影响。
不敏感的模型可能会忽略非本质的自我矛盾，而过度敏感的模型可能会将微小的不一致误解为矛盾。为了解决这个问题，我们引入了一个任务，旨在通过给出前k个证据文本来检测自我矛盾。
虽然我们的创作过程引入的自相矛盾被认为是文档中最明显的错误，但它应该出现在模型提供的前k个证据文本中。我们仅在CONTRADOC-POS上进行了测试。
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
# 保持 LLM 配置不变
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

try:
    # 增加 max_tokens 以便模型有空间返回 k 个证据文本和解释
    llm = OpenAIClient(
        model="gpt-4.1-mini",
        temperature=0.1,
        top_p=0.9,
        max_tokens=1500  # 恢复到 1500，确保输出空间
    )
except Exception as e:
    print(f"Error initializing OpenAIClient: {e}. Exiting.")
    exit()


# --- 2. 核心检测函数 (已修改为 Top-K 证据定位) ---

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
        # 清理并解析 JSON
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text)  # 不转换为小写，因为文本证据需要保持大小写

        evidence_list = response_json.get("top_k_evidence", [])
        explanation = response_json.get("explanation", default_result["explanation"]).strip().strip('"').strip("'")

        # 确保 evidence_list 是列表且元素是字符串
        if not isinstance(evidence_list, list) or not all(isinstance(e, str) for e in evidence_list):
            evidence_list = []

        return {"top_k_evidence": evidence_list[:k], "explanation": explanation}

    except Exception:
        return default_result


# --- 3. 数据集加载和抽样函数 (仅 Positives) ---

def load_and_sample_contradoc_data_pos(full_data: Dict, num_pos: int = 40) -> List[Dict[str, Any]]:
    """
    仅从 ContraDoc-POS 结构中加载数据，并按照要求进行抽样。
    - 正例 (pos): 尽量覆盖不同的 'contra_type'，并保留 'evidence' 字段。
    """
    pos_data = full_data.get("pos", {})
    pos_list = []

    # 按照 contra_type 分组
    pos_grouped = collections.defaultdict(list)
    for unique_id, doc_info in pos_data.items():
        types = doc_info.get("contra_type", ["Unknown"])
        for c_type in types:
            # 必须包含真实的 'evidence' 用于后续匹配
            pos_grouped[c_type].append({
                "unique_id": unique_id,
                "text": doc_info["text"],
                "true_evidence": doc_info.get("evidence", ""),  # 真实矛盾证据
                "type": c_type
            })

    # 抽样逻辑 (保持类型均衡)
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
    random.shuffle(pos_list)

    print(f"\n--- 数据抽样统计 (仅 Positives) ---")
    print(f"正例 (矛盾) 抽取数: {len(pos_list)} / 目标 {num_pos}")

    return pos_list


# --- 4. 评估主函数 (Top-K 评估) ---

def evaluate_top_k_detection(dataset: List[Dict[str, Any]], llm_client: Any, system_prompt: str, k: int = 3):
    """
    对数据集执行 Top-K 矛盾证据检测并计算成功率。
    """
    total_samples = len(dataset)
    successful_detections = 0
    detailed_results = []

    model_name = getattr(llm_client, 'model', 'MockLLM')
    print(f"\n--- 开始 Top-{k} 证据检测评估 ({total_samples} 个文档, 模型: {model_name}) ---")

    for i, item in enumerate(dataset):
        doc_text = item["text"]
        unique_id = item["unique_id"]
        true_evidence = item["true_evidence"].strip()

        # 获取模型的预测证据列表和解释
        model_output = detect_top_k_evidence_with_gpt(llm_client, doc_text, system_prompt, k)
        predicted_evidence = model_output["top_k_evidence"]
        explanation = model_output["explanation"]

        # 检查真实证据是否在 Top-K 预测证据中
        is_hit = False

        # 使用模糊匹配检查，因为模型可能返回稍微不同但意义相同的片段
        # 简单方法: 检查真实证据是否是预测证据的子串，或反之

        # 简化匹配逻辑：检查真实证据是否包含在任何一个预测片段中
        for pred_span in predicted_evidence:
            # 使用 SequenceMatcher 进行模糊匹配，阈值设为 0.8
            if difflib.SequenceMatcher(None, true_evidence.lower(), pred_span.lower()).ratio() >= 0.8 or \
                    true_evidence.lower() in pred_span.lower() or \
                    pred_span.lower() in true_evidence.lower():
                is_hit = True
                break

        if is_hit:
            successful_detections += 1

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
        print(f"[{status_symbol}] ID: {unique_id} | 进度: {i + 1}/{total_samples}")

        if (i + 1) % 10 == 0 or i == total_samples:
            print(f"当前 Top-{k} 成功率: {successful_detections / (i + 1):.4f}")
            print("-" * 20)

    # 计算最终指标
    success_rate = successful_detections / total_samples if total_samples > 0 else 0

    print("\n--- 评估结果 (Metrics) ---")
    print(f"Top-{k} 证据检测成功数: {successful_detections} / {total_samples}")
    print(f"Top-{k} 成功率 (Success Rate): {success_rate:.4f}")

    # 准备保存的指标字典
    metrics_to_save = {
        "Model": model_name,
        "Total_Samples": total_samples,
        f"Top_{k}_Success_Rate": f"{success_rate:.4f}",
        "Hit_Count": successful_detections
    }

    # 保存结果到文件
    save_top_k_results_to_file(metrics_to_save, detailed_results, model_name, k)


# --- 5. 结果保存辅助函数 (为 Top-K 调整) ---

def save_top_k_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str, k: int):
    """保存 Top-K 评估指标和详细的文档结果到本地文件。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"ContraDoc_Top{k}_Eval_{model_name}_{timestamp}"

    # 1. 保存详细结果 (JSON)
    json_path = f"{base_filename}_details.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"\n✨ 详细评估结果已保存到: {json_path}")

    # 2. 保存指标结果 (TXT)
    txt_path = f"{base_filename}_metrics.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ContraDoc Top-{k} 证据检测评估指标 ({model_name}) ---\n")
        f.write(f"时间: {timestamp}\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("-" * 40 + "\n")

    print(f"✨ 最终评估指标已保存到: {txt_path}")


# --- 6. 执行脚本 ---

if __name__ == "__main__":

    # 设置 K 值
    TOP_K = 3
    SAMPLES = 40  # 仅在 pos 上抽样 40 篇

    try:
        with open("../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json", 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print("Error: ContraDoc.json not found. Please check the file path.")
        exit()
    except json.JSONDecodeError:
        print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
        exit()

    # 仅从 POSitives 中抽样
    contradoc_dataset = load_and_sample_contradoc_data_pos(full_data, num_pos=SAMPLES)

    # 定义系统提示 (用于 Top-K 定位)
    SYSTEM_PROMPT = (
        "You are an expert natural language processing model specialized in identifying the specific spans of text "
        "that cause self-contradiction within a single document. "
        "Your task is to analyze the document and return the most crucial text spans that contribute to the contradiction."
    )

    # 执行 Top-K 评估
    evaluate_top_k_detection(contradoc_dataset, llm, SYSTEM_PROMPT, k=TOP_K)