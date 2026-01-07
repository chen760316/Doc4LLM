# DSCD_knowledge_conflict_GPT_Inference.py
"""
使用的数据集：knowledge_conflict
使用的基准："gpt-4.1-mini"
"""
from knowledge_graph_maker import OpenAIClient
from datasets import Dataset
from sklearn.metrics import classification_report
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# --- 0. 文件路径和配置 ---
REPORT_DIR = "report"
REPORT_FILE_PATH = os.path.join(REPORT_DIR, "knowledge_conflict_gpt_report.txt")
# 假设您的文件路径是这个
KNOWLEDGE_CONFLICT_FILE_PATH = "../../../knowledge conflict/sampled_data.json"
MAX_CONCURRENT_CALLS = 10  # 定义最大并发数
LIMIT = 100  # 限制样本数量，如果为 None 或 0 则使用全部

# --- 1. 配置和 LLM 客户端 ---
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

try:
    llm = OpenAIClient(
        model="gpt-4.1-mini",
        temperature=0.1,
        top_p=0.9,
        max_tokens=500
    )
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize OpenAIClient: {e}", file=sys.stderr)
    sys.exit(1)

# --- 2. 专门用于 DSCD 的系统提示 ---
# 此处的提示词需要针对 "知识冲突" 场景进行微调，使其更通用
CONFLICT_SYSTEM_PROMPT = (
    "You are an expert in detecting contradictions within a single text passage, which may contain multiple segments.\n"
    "Your task is to analyze the entire passage and determine if it contains any self-contradictory information, "
    "where two or more statements are logically incompatible with each other.\n"
    "Respond with a single JSON object with two keys:\n"
    '{"is_contradictory": <boolean>, "reason": "<brief explanation of the conflict, or why none was found>"}\n'
    "Respond with only the JSON object—no explanations, no markdown, no extra text."
)


# --- 3. 数据加载和映射函数 (针对 knowledge_conflict 结构重写) ---

def load_and_map_knowledge_conflict(file_path: str) -> Optional[Dataset]:
    """
    加载 knowledge_conflict JSON 文件 (列表结构) 并映射为 DSCD 任务的格式。

    原始数据样本格式：
    {
      "main_entity": "...",
      "parametric_knowledge": "...",
      "conflicting_knowledge": "...",
      "question_about_conflicting_segments": "...",
      "question_about_nonconflicting_segments": "...",
      "conflicting_texts": [...] # 包含问题和两个知识段
    }

    映射规则：
    - 冲突样本 (Label 1): question_about_conflicting_segments + parametric_knowledge + conflicting_knowledge
    - 非冲突样本 (Label 0): question_about_nonconflicting_segments + parametric_knowledge
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        if not isinstance(data_list, list):
            raise ValueError("File content is not a list.")
    except FileNotFoundError:
        print(f"FATAL ERROR: knowledge_conflict file not found at {file_path}", file=sys.stderr)
        return None
    except Exception as load_err:
        print(f"FATAL ERROR: Failed to load or parse file: {load_err}", file=sys.stderr)
        return None

    flattened_list = []

    for i, example_data in enumerate(data_list):
        unique_id = example_data.get('main_entity', f'ID_{i}')

        pk = example_data.get('parametric_knowledge', "")
        ck = example_data.get('conflicting_knowledge', "")
        qq_c = example_data.get('question_about_conflicting_segments', "")
        qq_nc = example_data.get('question_about_nonconflicting_segments', "")

        # 1. 构造冲突样本 (Label 1)
        # 将问题、原始知识和冲突知识拼接起来，形成一个包含矛盾信息的文档
        doc_conflict = " ".join([qq_c, pk, ck])
        flattened_list.append({
            'document': doc_conflict,
            'label': 1,  # 1: 冲突 (Contradictory)
            'original_id': f"{unique_id}_C"
        })

        # 2. 构造非冲突样本 (Label 0)
        # 将非冲突问题和原始知识拼接起来，形成一个不包含矛盾信息的文档
        doc_non_conflict = " ".join([qq_nc, pk])
        flattened_list.append({
            'document': doc_non_conflict,
            'label': 0,  # 0: 非冲突 (Non-Contradictory)
            'original_id': f"{unique_id}_NC"
        })

    print(f"原始 {len(data_list)} 条记录扩展为 {len(flattened_list)} 条 DSCD 样本 (1:1 冲突/非冲突)")
    return Dataset.from_list(flattened_list)


# --- 4. GPT API 调用和解析函数 (保持不变) ---

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
        return 0


# --- 5. 多线程评估函数 (保持不变) ---

def evaluate_multithreaded(test_dataset: Dataset, llm_client: OpenAIClient, system_prompt: str, max_workers: int, limit: Optional[int]):
    """使用多线程对测试集进行 GPT 零样本冲突检测"""

    if limit is not None and limit > 0:
        test_dataset = test_dataset.select(range(min(limit, len(test_dataset))))

    test_set_size = len(test_dataset)
    true_labels = test_dataset["label"]
    predicted_labels = [0] * test_set_size

    print(f"🚀 开始多线程推理 ({max_workers} 个并发线程) on {test_set_size} samples...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

            if completed_count % max(1, test_set_size // 10) == 0 or completed_count == test_set_size:
                print(f"✅ 已完成 {completed_count}/{test_set_size} 个文档...")

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

    os.makedirs(REPORT_DIR, exist_ok=True)

    print(f"🌍 正在加载 knowledge_conflict 数据集...")
    # 使用新的加载函数和路径变量
    raw_dataset = load_and_map_knowledge_conflict(KNOWLEDGE_CONFLICT_FILE_PATH)

    if raw_dataset is None:
        sys.exit(1)

    total_samples = len(raw_dataset)

    # 划分测试集 (使用 20% 作为测试集)
    train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset_full = train_test_split['test']

    eval_size = min(LIMIT, len(test_dataset_full)) if LIMIT else len(test_dataset_full)

    print(f"数据集总大小: {total_samples}")
    print(f"测试集原始大小 (20%): {len(test_dataset_full)}")
    print(f"实际评估样本数 (LIMIT={LIMIT}): {eval_size}")
    print(f"评估结果将被写入文件: {REPORT_FILE_PATH}")

    try:
        with open(REPORT_FILE_PATH, 'w', encoding='utf-8') as f:
            original_stdout = sys.stdout
            sys.stdout = f

            print(f"--- knowledge_conflict Zero-Shot Contradiction Detection Report (gpt-4.1-mini) ---")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Samples: {total_samples}, Evaluated Samples: {eval_size}")
            print("-" * 60)

            evaluate_multithreaded(
                test_dataset_full,
                llm,
                CONFLICT_SYSTEM_PROMPT,
                max_workers=MAX_CONCURRENT_CALLS,
                limit=LIMIT
            )

    except Exception as e:
        sys.stdout = original_stdout
        print(f"FATAL ERROR: An error occurred during evaluation or file writing: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        sys.stdout = original_stdout
        print(f"\n✅ 评估完成。详细报告已成功保存到 {REPORT_FILE_PATH}")