"""
测试gpt系列
"""
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import random # 保持导入，但不再使用
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
import collections
# 引入多线程库
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading  # 仅用于调试或高级控制

# --- 1. 初始化和配置 (保持不变) ---
# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-4o",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


# --- 2. 核心检测函数 (保持不变) ---

def detect_conflict_with_gpt(llm_client: Any, document_text: str, system_prompt: str) -> Tuple[int, str]:
    """
    使用 GPT 模型检测文档内冲突，返回 (标签, 解释)
    标签: 1 for YES (矛盾), 0 for NO (非矛盾)
    解释: 模型的文字解释
    """

    # 针对 Yes/No 任务，强制要求 JSON 输出，包含解释字段
    system_prompt_json = (
        f"{system_prompt}. "
        "Your output MUST be a single JSON object: { \"is_contradictory\": true/false, \"explanation\": \"[Your reason or evidence for the decision.]\" }. "
        "Do not include any other text outside the JSON block. Output only the JSON."
    )

    default_explanation = "无法获取模型响应或 JSON 解析失败。"

    try:
        # 在多线程环境中调用 API
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt_json)

    except TypeError:
        return 0, default_explanation
    except Exception:
        # 捕捉 API 错误、速率限制等
        return 0, default_explanation

    try:
        # 清理并解析 JSON
        response_text = response_text.strip().replace("```json", "").replace("```", "").replace("`", "").strip()
        response_json = json.loads(response_text.lower())

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


# --- 3. 完整加载数据集函数 (取代原采样函数) ---

def load_all_contradoc_data(full_data: Dict) -> List[Dict[str, Any]]:
    """
    从 ContraDoc JSON 结构中加载所有正例 (pos) 和负例 (neg) 数据，不进行采样。
    """
    pos_data = full_data.get("pos", {})
    neg_data = full_data.get("neg", {})

    all_data_list = []

    # 1. 加载所有正例 (Positives)
    for unique_id, doc_info in pos_data.items():
        # 正例标签为 1
        item = {
            "unique_id": unique_id,
            "text": doc_info["text"],
            "label": 1
        }
        # 如果需要，保留 contra_type 信息
        if "contra_type" in doc_info:
            item["type"] = doc_info["contra_type"]

        all_data_list.append(item)

    # 2. 加载所有负例 (Negatives)
    for unique_id, doc_info in neg_data.items():
        # 负例标签为 0
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


# --- 4. 结果保存辅助函数 (保持不变) ---

def save_results_to_file(metrics: Dict, detailed_results: List[Dict], model_name: str):
    """保存评估指标和详细的文档结果到本地文件，保存至 report/binary_judge_report 目录下。"""

    # 定义目标目录
    SAVE_DIR = "report/binary_judge_report_gpt-4o"
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

def evaluate_binary_detection_multithreaded(dataset: List[Dict[str, Any]], llm_client: Any, system_prompt: str, max_workers: int = 10):
    """
    对数据集执行二元检测并计算评估指标，使用多线程并发调用 LLM API。
    """
    true_labels = []
    pred_labels = []
    detailed_results = []

    model_name = getattr(llm_client, 'model', 'MockLLM')
    print(f"\n--- 开始评估 ({len(dataset)} 个文档, 模型: {model_name}, 使用 {max_workers} 个线程) ---")

    # 用于存储 future 对象和它们对应的数据索引
    future_to_index = {}

    # 使用 ThreadPoolExecutor 实现并发
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, item in enumerate(dataset):
            doc_text = item["text"]
            # 提交任务到线程池
            future = executor.submit(detect_conflict_with_gpt, llm_client, doc_text, system_prompt)
            # 存储 future 和原始数据集的索引
            future_to_index[future] = i

        # 收集结果并按提交顺序（i）进行排序
        results_list = [None] * len(dataset)

        # 用于跟踪处理进度的计数器
        processed_count = 0

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

            # 将结果放入正确的位置，以便后续按序处理
            results_list[i] = (true_label, predicted_label, result_entry)
            processed_count += 1

            # 打印状态和结果 (可选，但有助于跟踪进度)
            if predicted_label == 1:
                print(f"\n*** ❗ 检测到矛盾 (ID: {unique_id}) ***")
                print(f"真实标签: {'矛盾 (1)' if true_label == 1 else '非矛盾 (0)'}")
                print(f"模型解释: {explanation}")

            print(f"✅ 已处理 {processed_count}/{len(dataset)} 个文档...")


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
        "Model": model_name,
        "Total_Samples": len(dataset),
        "Accuracy": f"{accuracy:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1_Score": f"{f1:.4f}",
        "classification_report": cls_report
    }

    # 保存结果到文件
    save_results_to_file(metrics_to_save, detailed_results, model_name)


# --- 6. 执行脚本 ---

if __name__ == "__main__":

    data_path = ""
    full_data = None

    # 尝试加载数据文件
    try:
        # 原始问题中给出的路径
        data_path = "../../../../ContraDOC/sampled_contradoc.json"
        with open(data_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        try:
            # 尝试备份路径
            data_path = "../../../../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"
            with open(data_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
        except FileNotFoundError:
            print(f"错误: ContraDoc.json not found. Please check paths: {data_path} and the hardcoded paths.")
            exit()
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {data_path}. Please check file formatting.")
            exit()
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {data_path}. Please check file formatting.")
        exit()

    # 检查数据是否成功加载
    if full_data is None:
        print("错误: 无法加载 ContraDoc 数据，评估终止。")
        exit()

    # 使用新的函数加载所有数据，不进行采样
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

    # 执行评估 (使用多线程版本)
    # 建议将 max_workers 设置为你拥有的 LLM API 密钥允许的最大并发数。
    MAX_CONCURRENT_CALLS = 10

    if len(contradoc_dataset) > 0:
        evaluate_binary_detection_multithreaded(
            contradoc_dataset,
            llm,
            SYSTEM_PROMPT,
            max_workers=MAX_CONCURRENT_CALLS
        )
    else:
        print("警告: 加载的数据集为空，评估未执行。")