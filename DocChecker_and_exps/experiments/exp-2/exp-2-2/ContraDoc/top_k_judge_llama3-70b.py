# -*- coding: utf-8 -*-
from openai import OpenAI, APIError
import json
from dotenv import load_dotenv
import os
import random
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import collections
import difflib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed  # 引入多线程库

# --- 1. 配置和 LLM 客户端 (使用 Llama/Groq 配置) ---
print("--- 1. 正在加载配置和初始化 Llama Client ---")
load_dotenv()

# 使用 Llama/Groq 相关的环境变量
API_KEY = os.getenv("Llama_API_KEY") # 假设环境变量名为 Llama_API_KEY
BASE_URL = os.getenv("Llama_BASE_URL", None) # 假设环境变量名为 Llama_BASE_URL
MODEL_NAME = "llama-3-70b"  # 指定使用的 Llama 模型名称

if not API_KEY or not BASE_URL:
    print("❌ 错误: Llama_API_KEY 或 Llama_BASE_URL 未在环境中设置！")
    exit()

print(f"✅ 找到 API Key (长度: {len(API_KEY)})")
print(f"✅ 找到 Base URL: {BASE_URL}")
print(f"✅ 使用模型: {MODEL_NAME}")

# LLM 客户端初始化，使用 OpenAI 兼容客户端 (适用于 Groq 等)
try:
    # 关键：使用 Llama 的 Base URL 和 Key 初始化 OpenAI Client
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    print(f"✅ Llama Client (通过 OpenAI 兼容接口) 初始化成功")
except Exception as e:
    print(f"❌ 错误: Client 初始化失败: {e}")
    exit()

# 定义结果保存目录
SAVE_DIR = "report/top_k_judge_report_llama3-70b"


# --- 2. 核心检测函数 (Top-K 证据定位 - 使用 OpenAI 兼容客户端) ---

def detect_top_k_evidence_with_gpt(llm_client: OpenAI, document_text: str, system_prompt: str, k: int = 3) -> Dict[str, Any]:
    """
    使用 Llama 模型通过 OpenAI 兼容接口检测文档内冲突，返回 Top-K 证据和解释。
    """
    # 强制输出 JSON 的系统提示
    system_prompt_json = (
        f"{system_prompt}. "
        f"You must identify the {k} most likely text spans that, if true, would cause the document to be self-contradictory. "
        "Your output MUST be a single JSON object: "
        "{ \"top_k_evidence\": [\"span 1\", \"span 2\", ...], \"explanation\": \"[Your reason for selection.]\" }. "
        "Each span in 'top_k_evidence' must be an exact short quote from the document text. Output ONLY the JSON object, do NOT include any other text or markdown fences (e.g., ```json)."
    )

    default_result = {"top_k_evidence": [], "explanation": "无法获取模型响应或 JSON 解析失败。"}

    try:
        # 核心 API 调用，使用 OpenAI Client 的标准参数
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,  # 使用全局定义的 Llama 模型
            messages=[
                {"role": "system", "content": system_prompt_json},
                {"role": "user", "content": document_text}
            ],
            temperature=0.1,
            max_tokens=1500,
            timeout=30
        )

        if not response.choices or not response.choices[0].message:
            return default_result

        response_text = response.choices[0].message.content

    except APIError as e:
        # 捕获 API 错误
        print(f"API Error during call: {e}")
        return default_result
    except Exception:
        return default_result

    try:
        # 清理并解析 JSON
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text)

        evidence_list = response_json.get("top_k_evidence", [])
        explanation = response_json.get("explanation", default_result["explanation"]).strip().strip('"').strip("'")

        if not isinstance(evidence_list, list) or not all(isinstance(e, str) for e in evidence_list):
            evidence_list = []

        return {"top_k_evidence": evidence_list[:k], "explanation": explanation}

    except Exception:
        # JSON 解析或结构不符合预期
        print(f"JSON 解析失败的文本: {response_text[:200]}...")
        return default_result


# --- 3. 数据集加载函数 (加载所有 Positives, 不进行采样) (保持不变) ---

def load_all_contradoc_data_pos(full_data: Dict) -> List[Dict[str, Any]]:
    """
    仅从 ContraDoc-POS 结构中加载所有正例数据，不进行任何抽样。
    """
    pos_data = full_data.get("pos", {})
    pos_list = []

    for unique_id, doc_info in pos_data.items():
        # 确保 'evidence' 是字符串或列表，并将其处理为单个字符串
        evidence = doc_info.get("evidence", "")
        if isinstance(evidence, list):
            true_evidence = evidence[0] if evidence else ""
        else:
            true_evidence = str(evidence)

        # 尝试获取第一个 contra_type
        contra_type = doc_info.get("contra_type", ["Unknown"])
        doc_type = contra_type[0] if isinstance(contra_type, list) and contra_type else "Unknown"

        item = {
            "unique_id": unique_id,
            "text": doc_info["text"],
            "true_evidence": true_evidence,
            "type": doc_type
        }
        pos_list.append(item)

    random.shuffle(pos_list)

    print(f"\n--- 数据加载统计 (仅 Positives) ---")
    print(f"正例 (矛盾) 总加载数: {len(pos_list)}")

    return pos_list


# --- 4. 评估主函数 (Top-K 评估 - 多线程版本) (保持不变) ---

def evaluate_top_k_detection_multithreaded(dataset: List[Dict[str, Any]], llm_client: OpenAI, system_prompt: str, k: int = 3, max_workers: int = 10):
    """
    对数据集执行 Top-K 矛盾证据检测并计算成功率，并以伪二元分类指标形式报告结果。
    """
    total_samples = len(dataset)
    model_name = MODEL_NAME  # 使用全局模型名
    print(f"\n--- 开始 Top-{k} 证据检测评估 ({total_samples} 个文档, 模型: {model_name}, 使用 {max_workers} 个线程) ---")

    future_to_index = {}
    start_time = datetime.now()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(dataset):
            doc_text = item["text"]
            # 提交任务
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
    true_labels = [1] * total_samples
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
    print(f"Precision (精确度): {precision:.4f}")
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


# --- 5. 结果保存辅助函数 (已更新为 Llama 路径) ---

def save_top_k_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str, k: int):
    """保存 Top-K 评估指标和详细的文档结果到本地文件，保存至 report/top_k_judge_report_llama 目录下。"""

    # 创建目录（如果不存在）
    os.makedirs(SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 更改文件名以反映 Llama 模型
    base_filename = f"ContraDoc_Top{k}_Eval_{model_name.replace('-', '_')}_{timestamp}"

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
    MAX_CONCURRENT_CALLS = 10  # 并发线程数

    try:
        data_path = "../../../../ContraDOC/sampled_contradoc.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print("Error: ContraDoc.json not found. Please check the file path.")
        exit()
    except json.JSONDecodeError:
        print("Error: Failed to decode ContraDOC.json. Please check file formatting.")
        exit()

    # 加载所有正例数据 (不采样)
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
            client,  # 使用全局创建的 Llama 客户端
            SYSTEM_PROMPT,
            k=TOP_K,
            max_workers=MAX_CONCURRENT_CALLS
        )
    else:
        print("警告: 数据集中没有找到正例 (矛盾) 文档，评估未执行。")