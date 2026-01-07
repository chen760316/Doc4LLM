# -*- coding: utf-8 -*-
from openai import OpenAI, APIError
import json
from dotenv import load_dotenv
import os
import random
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# --- 1. 初始化和配置 (使用 Llama 3/Groq 配置) ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# 使用 Llama 3/Groq 相关的环境变量
GROQ_API_KEY = os.getenv("Llama_API_KEY")
GROQ_BASE_URL = os.getenv("Llama_BASE_URL", None)
LLAMA3_MODEL = "llama-2-70b"  # Groq 支持的模型

if not GROQ_API_KEY or not GROQ_BASE_URL:
    # 注意：如果 BASE_URL 为 None 且 API 密钥是 Groq 官方的，OpenAI 客户端可能会自动使用默认 Groq URL。
    # 但为保证通过中转站的兼容性，我们要求 BASE_URL 存在。
    print("❌ 错误: Llama_API_KEY 或 Llama_BASE_URL 未在环境中设置！")
    exit()

print(f"✅ 找到 API Key (长度: {len(GROQ_API_KEY)})")
print(f"✅ 找到 Base URL: {GROQ_BASE_URL}")
print(f"✅ 使用模型: {LLAMA3_MODEL}")

# LLM 客户端初始化，使用 OpenAI 兼容客户端
try:
    # 初始化 OpenAI Client，连接到 Groq/中转站
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL
    )
    print("✅ Llama 3 Client (通过 OpenAI 兼容接口) 初始化成功")
except Exception as e:
    print(f"❌ 错误: Client 初始化失败: {e}")
    exit()


# --- 2. 核心检测函数 (保持不变，但使用 LLAMA3_MODEL) ---

def detect_conflict_with_llm(llm_client: OpenAI, model_name: str, document_text: str, system_prompt: str) -> Tuple[int, str]:
    """
    使用 LLM 模型通过 OpenAI 兼容接口检测文档内冲突，返回 (标签, 解释)
    """
    system_prompt_json = (
        f"{system_prompt}. "
        "Your output MUST be a single JSON object: { \"is_contradictory\": true/false, \"explanation\": \"[Your reason or evidence for the decision.]\" }. "
        "Do not include any other text outside the JSON block. Output only the JSON."
    )

    default_explanation = "无法获取模型响应或 JSON 解析失败。"

    try:
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt_json},
                {"role": "user", "content": document_text}
            ],
            temperature=0.1,
            max_tokens=1500,
            timeout=30.0
        )

        if not response.choices or not response.choices[0].message:
            return 0, default_explanation

        response_text = response.choices[0].message.content

    except APIError as e:
        return 0, f"API 调用失败: {e}"
    except Exception:
        return 0, default_explanation

    try:
        response_text = response_text.strip().replace("```json", "").replace("```", "").replace("`", "").strip()
        response_json = json.loads(response_text.lower())

        is_contradictory = response_json.get("is_contradictory", False)
        if isinstance(is_contradictory, str):
            label = 1 if is_contradictory == "true" else 0
        else:
            label = int(is_contradictory)

        explanation = response_json.get("explanation", default_explanation).strip().strip('"').strip("'")

        return label, explanation

    except Exception:
        return 0, default_explanation


# --- 3. 数据加载函数 (加载全部数据，保持不变) ---

def load_all_contradoc_data(full_data: Dict) -> List[Dict[str, Any]]:
    """
    从 ContraDoc JSON 结构中加载所有正例 (pos) 和负例 (neg) 数据，不进行采样。
    """
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    all_data_list = []

    # 1. 加载所有正例 (Positives)
    for unique_id, doc_info in pos_data.items():
        item = {
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 1
        }
        if "contra_type" in doc_info:
            item["type"] = doc_info["contra_type"]

        all_data_list.append(item)

    # 2. 加载所有负例 (Negatives)
    for unique_id, doc_info in neg_data.items():
        all_data_list.append({
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 0
        })

    print(f"\n--- 数据加载统计 ---")
    print(f"正例 (矛盾) 总加载数: {len(pos_data)}")
    print(f"负例 (非矛盾) 总加载数: {len(neg_data)}")
    print(f"评估总样本数: {len(all_data_list)}")

    # 3. 随机打乱并返回
    random.shuffle(all_data_list)
    return all_data_list


# --- 4. 结果保存辅助函数 (修改保存目录) ---

def save_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存评估指标和详细的文档结果到本地文件，保存至 report/binary_judge_report_llama3 目录下。"""

    # 定义目标目录
    SAVE_DIR = "report/binary_judge_report_llama3"
    # 创建目录（如果不存在）
    os.makedirs(SAVE_DIR, exist_ok=True)

    base_filename = f"ContraDoc_Binary_Eval_{model_name}"

    # 1. 保存详细结果 (JSON)
    json_path = os.path.join(SAVE_DIR, f"{base_filename}_details.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"\n✨ 详细评估结果已保存到: {json_path}")

    # 2. 保存指标结果 (TXT)
    txt_path = os.path.join(SAVE_DIR, f"{base_filename}_metrics.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- ContraDoc 二元检测评估指标 ({model_name}) ---\n")
        f.write("-" * 40 + "\n")
        # 写入核心指标
        f.write(f"Accuracy (准确性): {metrics.get('Accuracy', 'N/A')}\n")
        f.write(f"Precision (精确度): {metrics.get('Precision', 'N/A')}\n")
        f.write(f"Recall (召回率): {metrics.get('Recall', 'N/A')}\n")
        f.write(f"F1 Score (F1 分数): {metrics.get('F1_Score', 'N/A')}\n")
        f.write("-" * 40 + "\n")
        f.write("详细分类报告:\n")
        f.write(metrics.get("classification_report", "N/A"))

    print(f"✨ 最终评估指标已保存到: {txt_path}")


# --- 5. 评估主函数 (保持不变) ---

def evaluate_binary_detection_multithreaded(dataset: List[Dict[str, Any]], llm_client: OpenAI, model_name: str, system_prompt: str, max_workers: int = 10):
    """
    对数据集执行二元检测并计算评估指标，使用多线程并发调用 LLM API。
    """
    true_labels = []
    pred_labels = []
    detailed_results = []

    print(f"\n--- 开始评估 ({len(dataset)} 个文档, 模型: {model_name}, 使用 {max_workers} 个线程) ---")

    future_to_index = {}

    lock = threading.Lock()
    processed_count = 0
    total_samples = len(dataset)

    # 使用 ThreadPoolExecutor 实现并发
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(dataset):
            doc_text = item["text"]
            # 提交任务到线程池，注意传递 model_name
            future = executor.submit(detect_conflict_with_llm, llm_client, model_name, doc_text, system_prompt)
            future_to_index[future] = i

        results_list = [None] * total_samples

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            item = dataset[i]
            unique_id = item["unique_id"]
            true_label = item["label"]

            try:
                predicted_label, explanation = future.result()
            except Exception as exc:
                print(f"线程执行任务 {unique_id} 时发生异常: {exc}")
                predicted_label, explanation = 0, f"执行异常: {exc}"

            result_entry = {
                "unique_id": unique_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "explanation": explanation,
                "is_correct": (true_label == predicted_label)
            }

            results_list[i] = (true_label, predicted_label, result_entry)

            with lock:
                processed_count += 1
                if predicted_label == 1:
                    print(f"\n*** ❗ 检测到矛盾 (ID: {unique_id}) - 进度: {processed_count}/{total_samples} ***")
                    print(f"真实标签: {'矛盾 (1)' if true_label == 1 else '非矛盾 (0)'}")
                    print(f"模型解释: {explanation}")
                else:
                    print(f"✅ 已处理 {processed_count}/{total_samples} 个文档...")

    # --- 结果汇总和指标计算 ---
    for true_label, predicted_label, result_entry in results_list:
        true_labels.append(true_label)
        pred_labels.append(predicted_label)
        detailed_results.append(result_entry)

    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    cls_report = classification_report(true_labels, pred_labels, target_names=["Non-Contradictory (0)", "Contradictory (1)"], zero_division=0)

    print("\n--- 评估结果 (Metrics) ---")
    print(f"Accuracy (准确性): {accuracy:.4f}")
    print(f"Precision (精确度): {precision:.4f}")
    print(f"Recall (召回率): {recall:.4f}")
    print(f"F1 Score (F1 分数): {f1:.4f}")
    print("\n--- 详细分类报告 ---")
    print(cls_report)

    metrics_to_save = {
        "Model": model_name,
        "Total_Samples": len(dataset),
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1_Score": f"{f1:.4f}",
        "classification_report": cls_report
    }

    save_results_to_file(metrics_to_save, detailed_results, model_name)


# --- 6. 执行脚本 ---

if __name__ == "__main__":

    # 尝试加载数据
    try:
        data_path = "../../../../ContraDOC/sampled_contradoc.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        try:
            data_path = "../../../../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"
            with open(data_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: ContraDoc.json not found. Please check paths.")
            exit()
        except json.JSONDecodeError:
            print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
            exit()
    except json.JSONDecodeError:
        print("Error: Failed to decode ContraDoc.json. Please check file formatting.")
        exit()

    if full_data is None:
        print("错误: 无法加载 ContraDoc 数据，评估终止。")
        exit()

    # 使用加载所有数据的函数，不进行采样
    contradoc_dataset = load_all_contradoc_data(full_data)

    # 定义系统提示
    SYSTEM_PROMPT = (
        "You are an expert natural language processing model specialized in identifying self-contradictory "
        "statements within a single document. Carefully read the provided document. "
        "Determine if the document contains any internal contradictions, meaning two or more sentences "
        "in the text present facts or information that cannot simultaneously be true. "
        "If it is self-contradictory, output true. Otherwise, output false. "
        "The 'explanation' field must state the contradictory evidence or the reason for the non-contradictory decision."
    )

    # 执行评估 (使用 Llama 3/Groq 模型)
    MAX_CONCURRENT_CALLS = 10  # Groq 通常支持更高的并发数

    if len(contradoc_dataset) > 0:
        evaluate_binary_detection_multithreaded(
            contradoc_dataset,
            client,  # 使用全局创建的 OpenAI 兼容客户端
            LLAMA3_MODEL,  # 传递 Llama 3 模型名称
            SYSTEM_PROMPT,
            max_workers=MAX_CONCURRENT_CALLS
        )
    else:
        print("警告: 加载的数据集为空，评估未执行。")