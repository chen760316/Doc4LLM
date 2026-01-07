# DSCD_GPT_Inference.py (Multithreaded Version with File Output)
"""
使用的数据集：ContraDOC
使用的基准："gpt-4.1-mini"
"""
from knowledge_graph_maker import OpenAIClient
from datasets import Dataset
import json
from dotenv import load_dotenv
import os
from typing import Dict, List, Any
from sklearn.metrics import classification_report
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed  # 引入多线程库
from datetime import datetime

# --- 0. 文件路径定义 ---
REPORT_DIR = "report"
REPORT_FILE_PATH = os.path.join(REPORT_DIR, "ContraDOC_gpt_report.txt")

# --- 1. 配置和 LLM 客户端 ---
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

try:
    # 尝试初始化 LLM 客户端
    # ⚠️ 注意：如果 knowledge_graph_maker 无法导入，请确保环境正确或使用模拟客户端。
    llm = OpenAIClient(
        model="gpt-4.1-mini",
        temperature=0.1,
        top_p=0.9,
        max_tokens=500
    )
except Exception as e:
    # 如果初始化失败，程序将退出，确保不会在未初始化的状态下运行
    print(f"FATAL ERROR: Failed to initialize OpenAIClient: {e}", file=sys.stderr)
    sys.exit(1)

# --- 2. 专门用于 DSCD 的系统提示 ---
CONFLICT_SYSTEM_PROMPT = (
    "You are an expert in detecting contradictions within a single document.\n"
    "Your task is to analyze the entire document and determine if it contains any self-contradictory information, "
    "where two or more statements are logically incompatible with each other.\n"
    "Respond with a single JSON object with two keys:\n"
    '{"is_contradictory": <boolean>, "reason": "<brief explanation of the conflict, or why none was found>"}\n'
    "Respond with only the JSON object—no explanations, no markdown, no extra text."
)


# --- 3. 数据加载函数 (展平 ContraDOC 结构) ---
def load_contradoc_data(file_path: str) -> List[Dict]:
    """手动加载 ContraDOC JSON 文件并展平结构"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: ContraDOC file not found at {file_path}", file=sys.stderr)
        return []

    flattened_data = []
    # 处理 "pos" (有矛盾, label=1)
    if "pos" in data:
        for unique_id, doc_data in data["pos"].items():
            flattened_data.append({
                "document": doc_data["text"],
                "label": 1,
                "unique_id": unique_id
            })

    # 处理 "neg" (无矛盾, label=0)
    if "neg" in data:
        for unique_id, doc_data in data["neg"].items():
            flattened_data.append({
                "document": doc_data["text"],
                "label": 0,
                "unique_id": unique_id
            })

    return flattened_data


# --- 4. GPT API 调用和解析函数 ---
def detect_conflict_with_gpt(llm_client: OpenAIClient, document_text: str, system_prompt: str) -> int:
    """使用 GPT 模型检测文档内冲突，并返回整数标签 (0 或 1)"""
    try:
        response_text = llm_client.generate(user_message=document_text, system_message=system_prompt)
        response_text = response_text.strip().replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_text)

        is_contradictory = response_json.get("is_contradictory")
        if isinstance(is_contradictory, str):
            return 1 if is_contradictory.lower() == 'true' else 0
        elif isinstance(is_contradictory, bool):
            return int(is_contradictory)

        return 0

    except Exception:
        # 统一返回 0 (非矛盾) 作为失败时的默认值
        return 0


# --- 5. 多线程评估函数 ---
def evaluate_multithreaded(test_dataset: Dataset, llm_client: OpenAIClient, system_prompt: str, max_workers: int = 10):
    """使用多线程对测试集进行 GPT 零样本冲突检测"""
    test_set_size = len(test_dataset)
    true_labels = test_dataset["label"]
    predicted_labels = [0] * test_set_size

    print(f"🚀 开始多线程推理 ({max_workers} 个并发线程)...")

    # 使用 ThreadPoolExecutor 实现并发调用
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(detect_conflict_with_gpt, llm_client, example['document'], system_prompt): i
            for i, example in enumerate(test_dataset)
        }

        completed_count = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            completed_count += 1

            try:
                prediction = future.result()
                predicted_labels[index] = prediction
            except Exception:
                predicted_labels[index] = 0

            # 实时打印进度 (此进度会被写入文件，可以注释掉以减少文件写入量)
            # if completed_count % max(1, test_set_size // 10) == 0 or completed_count == test_set_size:
            #      print(f"✅ 已完成 {completed_count}/{test_set_size} 个文档...")

    # --- 评估结果 (输出到文件) ---
    print("\n--- 📊 GPT 零样本冲突检测结果 ---")

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=['Non-Contradictory (0)', 'Contradictory (1)'],
        digits=4
    )
    print("详细分类报告:")
    print(report)
    print(f"\n✅ 评估完成。总共处理了 {test_set_size} 条测试样本。")


# --- 6. 主执行逻辑 ---
if __name__ == '__main__':
    MAX_CONCURRENT_CALLS = 20
    data_files = {
        "train": "../../../ContraDOC/sampled_contradoc.json"
    }

    # 检查数据文件是否存在
    if not os.path.exists(data_files["train"]):
        print(f"FATAL ERROR: Data file not found at {data_files['train']}", file=sys.stderr)
        sys.exit(1)

    # 1. 创建报告目录
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 2. 加载数据
    flattened_data = load_contradoc_data(data_files["train"])
    if not flattened_data:
        sys.exit(1)

    raw_dataset = Dataset.from_list(flattened_data)
    total_samples = len(raw_dataset)

    # 3. 划分测试集
    train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = train_test_split['test']
    test_set_size = len(test_dataset)

    # 4. 设置输出重定向
    print(f"🌍 正在加载 ContraDOC 数据集并展平结构...")
    print(f"数据集总大小: {total_samples}")
    print(f"测试集大小 (20%): {test_set_size}")
    print(f"评估结果将被写入文件: {REPORT_FILE_PATH}")

    try:
        with open(REPORT_FILE_PATH, 'w', encoding='utf-8') as f:
            # 备份原始 stdout
            original_stdout = sys.stdout
            sys.stdout = f  # 重定向 stdout 到文件

            # 打印起始信息到文件
            print(f"--- ContraDOC Zero-Shot Contradiction Detection Report (gpt-4.1-mini) ---")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Samples: {total_samples}, Test Set Size: {test_set_size}")
            print("-" * 60)

            # 执行多线程评估
            evaluate_multithreaded(
                test_dataset,
                llm,
                CONFLICT_SYSTEM_PROMPT,
                max_workers=MAX_CONCURRENT_CALLS
            )

    except Exception as e:
        # 确保在出错时恢复 stdout
        sys.stdout = original_stdout
        print(f"FATAL ERROR: An error occurred during evaluation or file writing: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        # 5. 恢复 stdout
        sys.stdout = original_stdout
        print(f"\n✅ 评估完成。详细报告已成功保存到 {REPORT_FILE_PATH}")