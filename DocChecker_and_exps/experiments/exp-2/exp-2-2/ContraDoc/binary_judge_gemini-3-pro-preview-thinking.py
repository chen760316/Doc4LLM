# -*- coding: utf-8 -*-
from openai import OpenAI
from openai import APIError
import json
from dotenv import load_dotenv
import os
import random
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time # 引入 time 用于在多线程中打印进度

# --- 1. 初始化和配置 (使用 Gemini 配置) ---
load_dotenv()

# 使用 GEMINI 相关的环境变量
API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("GEMINI_BASE_URL", None)
MODEL_NAME = "gemini-3-pro-preview-thinking" # 指定使用的模型

if not API_KEY or not BASE_URL:
    print("❌ 错误: GEMINI_API_KEY 或 GEMINI_BASE_URL 未在环境中设置！")
    exit()

print(f"✅ 找到 API Key (长度: {len(API_KEY)})")
print(f"✅ 找到 Base URL: {BASE_URL}")
print(f"✅ 使用模型: {MODEL_NAME}")

# LLM 客户端初始化，使用 OpenAI 兼容客户端
try:
    # 初始化 OpenAI Client，连接到中转站
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    print("✅ Gemini Client (通过 OpenAI 兼容接口) 初始化成功")
except Exception as e:
    print(f"❌ 错误: Client 初始化失败: {e}")
    exit()


# --- 2. 核心检测函数 (已修改为使用 `openai.OpenAI` 客户端) ---

def detect_conflict_with_gpt(llm_client: OpenAI, document_text: str, system_prompt: str) -> Tuple[int, str]:
    """
    使用 Gemini 模型通过 OpenAI 兼容接口检测文档内冲突，返回 (标签, 解释)
    标签: 1 for YES (矛盾), 0 for NO (非矛盾)
    解释: 模型的文字解释
    """
    # 针对 Yes/No 任务，强制要求 JSON 输出
    system_prompt_json = (
        f"{system_prompt}. "
        "Your output MUST be a single JSON object: { \"is_contradictory\": true/false, \"explanation\": \"[Your reason or evidence for the decision.]\" }. "
        "Do not include any other text outside the JSON block. Output only the JSON."
    )

    default_explanation = "无法获取模型响应或 JSON 解析失败。"

    try:
        # 核心 API 调用，使用 OpenAI Client 的标准参数
        response = llm_client.chat.completions.create(
            model=MODEL_NAME, # 使用全局定义的 Gemini 模型
            messages=[
                {"role": "system", "content": system_prompt_json},
                {"role": "user", "content": document_text}
            ],
            temperature=0.1,
            max_tokens=1500,
        )

        if not response.choices or not response.choices[0].message:
            return 0, default_explanation

        response_text = response.choices[0].message.content

    except APIError as e:
        # 捕获 API 错误，如速率限制、无效密钥等
        return 0, f"API 调用失败: {e}"
    except Exception:
        # 捕捉其他未知错误
        return 0, default_explanation

    try:
        # 清理并解析 JSON
        # 需要移除 Markdown 块标记
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text.lower()) # 转换为小写以便解析 true/false 字符串

        # 尝试获取标签
        is_contradictory = response_json.get("is_contradictory", False)
        if isinstance(is_contradictory, str):
            label = 1 if is_contradictory == "true" else 0
        else:
            label = int(is_contradictory)

        # 尝试获取解释，并清理可能的引号
        explanation = response_json.get("explanation", default_explanation).strip().strip('"').strip("'")

        return label, explanation

    except Exception:
        # JSON 解析失败，返回默认值
        return 0, default_explanation


# --- 3. 模拟/加载数据集函数 (保持不变) ---

def load_and_sample_contradoc_data(full_data: Dict, num_pos: int = 30, num_neg: int = 30) -> List[Dict[str, Any]]:
    """从 ContraDoc JSON 结构中加载数据，并按照要求进行抽样。"""
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    # 1. 处理正例 (Positives)
    pos_list = []
    pos_grouped = collections.defaultdict(list)
    for unique_id, doc_info in pos_data.items():
        types = doc_info.get("contra_type", ["Unknown"])
        for c_type in types:
            pos_grouped[c_type].append({
                "unique_id": unique_id,
                "text": doc_info["text"],
                "label": 1,
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

    # 2. 处理负例 (Negatives)
    neg_available = []
    for unique_id, doc_info in neg_data.items():
        neg_available.append({
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 0
        })

    num_to_sample_neg = min(num_neg, len(neg_available))
    neg_list = random.sample(neg_available, num_to_sample_neg)

    print(f"\n--- 数据抽样统计 ---")
    print(f"正例 (矛盾) 抽取数: {len(pos_list)} / 目标 {num_pos}")
    print(f"负例 (非矛盾) 抽取数: {len(neg_list)} / 目标 {num_neg}")

    # 3. 合并和返回
    dataset = pos_list + neg_list
    random.shuffle(dataset)
    return dataset


# --- 4. 结果保存辅助函数 (保持不变) ---

def save_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存评估指标和详细的文档结果到本地文件，保存至 report/binary_judge_report 目录下。"""

    # 定义目标目录
    SAVE_DIR = "report/binary_judge_report_gemini-3-pro-preview-thinking"
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


# --- 5. 评估主函数 (保持不变，但使用新的客户端) ---

def evaluate_binary_detection_multithreaded(dataset: List[Dict[str, Any]], llm_client: OpenAI, system_prompt: str, max_workers: int = 10):
    """
    对数据集执行二元检测并计算评估指标，使用多线程并发调用 LLM API。
    """
    true_labels = []
    pred_labels = []
    detailed_results = []

    model_name = MODEL_NAME # 使用全局定义的模型名
    print(f"\n--- 开始评估 ({len(dataset)} 个文档, 模型: {model_name}, 使用 {max_workers} 个线程) ---")

    future_to_index = {}

    # 跟踪进度
    lock = threading.Lock()
    processed_count = 0
    total_samples = len(dataset)

    # 使用 ThreadPoolExecutor 实现并发
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(dataset):
            doc_text = item["text"]
            # 提交任务到线程池
            future = executor.submit(detect_conflict_with_gpt, llm_client, doc_text, system_prompt)
            # 存储 future 和原始数据集的索引
            future_to_index[future] = i

        # 收集结果并按提交顺序（i）进行排序
        results_list = [None] * total_samples

        for future in as_completed(future_to_index):
            i = future_to_index[future]
            item = dataset[i]
            unique_id = item["unique_id"]
            true_label = item["label"]

            try:
                # 获取线程执行的结果: (predicted_label, explanation)
                predicted_label, explanation = future.result()
            except Exception as exc:
                print(f"线程执行任务 {unique_id} 时发生异常: {exc}")
                predicted_label, explanation = 0, f"执行异常: {exc}"  # 默认安全值

            # 收集结果
            result_entry = {
                "unique_id": unique_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "explanation": explanation,
                "is_correct": (true_label == predicted_label)
            }

            # 将结果放入正确的位置
            results_list[i] = (true_label, predicted_label, result_entry)

            # 更新进度
            with lock:
                processed_count += 1
                if predicted_label == 1:
                    print(f"\n*** ❗ 检测到矛盾 (ID: {unique_id}) - 进度: {processed_count}/{total_samples} ***")
                    print(f"真实标签: {'矛盾 (1)' if true_label == 1 else '非矛盾 (0)'}")
                    print(f"模型解释: {explanation}")
                else:
                    print(f"✅ 已处理 {processed_count}/{total_samples} 个文档...")

    # --- 结果汇总和指标计算 ---

    # 遍历已排序的结果列表，填充 true_labels, pred_labels, detailed_results
    for true_label, predicted_label, result_entry in results_list:
        true_labels.append(true_label)
        pred_labels.append(predicted_label)
        detailed_results.append(result_entry)

    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    # 生成分类报告文本
    cls_report = classification_report(true_labels, pred_labels, target_names=["Non-Contradictory (0)", "Contradictory (1)"], zero_division=0)

    print("\n--- 评估结果 (Metrics) ---")
    print(f"Accuracy (准确性): {accuracy:.4f}")
    print(f"Precision (精确度): {precision:.4f}")
    print(f"Recall (召回率): {recall:.4f}")
    print(f"F1 Score (F1 分数): {f1:.4f}")
    print("\n--- 详细分类报告 ---")
    print(cls_report)

    # 准备保存的指标字典
    metrics_to_save = {
        "Model": MODEL_NAME, # 使用 Gemini 模型名
        "Total_Samples": len(dataset),
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1_Score": f"{f1:.4f}",
        "classification_report": cls_report
    }

    # 保存结果到文件
    save_results_to_file(metrics_to_save, detailed_results, MODEL_NAME)


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

    # 使用新的抽样函数加载数据集 (默认 30 正例, 30 负例)
    contradoc_dataset = load_and_sample_contradoc_data(full_data, num_pos=30, num_neg=30)

    # 定义系统提示
    SYSTEM_PROMPT = (
        "You are an expert natural language processing model specialized in identifying self-contradictory "
        "statements within a single document. Carefully read the provided document. "
        "Determine if the document contains any internal contradictions, meaning two or more sentences "
        "in the text present facts or information that cannot simultaneously be true. "
        "If it is self-contradictory, output true. Otherwise, output false. "
        "The 'explanation' field must state the contradictory evidence or the reason for the non-contradictory decision."
    )

    # 执行评估 (使用多线程版本)
    MAX_CONCURRENT_CALLS = 10
    evaluate_binary_detection_multithreaded(
        contradoc_dataset,
        client, # 使用全局创建的 OpenAI 兼容客户端
        SYSTEM_PROMPT,
        max_workers=MAX_CONCURRENT_CALLS
    )