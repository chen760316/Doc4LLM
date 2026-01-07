"""
使用的数据集：ECON
使用的基准："distilbert-base-cased-distilled-squad"
"""

import json
from typing import Dict, Any, List
from transformers import pipeline
from tqdm import tqdm # 用于显示进度条
import os

# --- 1. 配置 ---
# 文件路径占位符 - 请根据您的实际路径进行修改
ANSWER_CONFLICT_FILE = "../../../datasets/data_from_related_work/ECON/data/answer_conflicts/cwq-long-triples-alter_answer.json"
FACTOID_CONFLICT_FILE = "../../../datasets/data_from_related_work/ECON/data/factoid_conflicts/3_factoids/original/strategyqa_3facts_1mofified_0shuffle-test_set.json"
MAX_SAMPLES = 500 # 每个文件读取的最大样本数

# 新增：结果输出文件名
ANSWER_OUTPUT_FILE = "answer_conflict_results.json"
FACTOID_OUTPUT_FILE = "factoid_conflict_results.json"

# 加载问答模型 (保持不变)
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    tokenizer="distilbert-base-cased-distilled-squad"
)


# --- 2. 数据加载函数 (保持不变) ---

def load_econ_data(file_path: str, max_samples: int) -> List[Dict[str, Any]]:
    """从指定文件路径加载ECON数据集，并限制加载的样本数。"""
    data = []
    try:
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found at {file_path}. Please check the path.")
            return data

        with open(file_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            data = full_data[:max_samples]
            print(f"✅ Successfully loaded {len(data)} samples from: {file_path}")
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in {file_path}. Skipping.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading {file_path}: {e}")

    return data


# --- 3. 问答任务执行函数 (run_answer_conflict_qa, run_factoid_conflict_qa) ---
# 注意：这里需要确保将 float 转换加入到 run_answer_conflict_qa 中，以便 JSON 序列化。

def run_answer_conflict_qa(data: List[Dict[str, Any]], qa_func: Any):
    """处理答案冲突 (Answer Conflicts) 任务。"""
    results = []
    print("\n" + "=" * 80)
    print(f"## 🅰️ 运行答案冲突问答任务 ({len(data)} 样本)")
    print("=" * 80)

    for i, instance in enumerate(tqdm(data, desc="Answer Conflict")):
        q = instance["question"]

        if len(instance.get("texts", [])) < 2:
            continue

        context_A = instance["texts"][0]
        result_A = qa_func(question=q, context=context_A, top_k=1)

        context_B = instance["texts"][-1] if len(instance["texts"]) > 2 else instance["texts"][1]
        result_B = qa_func(question=q, context=context_B, top_k=1)

        context_Conflict = " ".join(instance["texts"])
        result_Conflict = qa_func(question=q, context=context_Conflict, top_k=1)

        # 记录结果，并确保分数是标准的 float 类型
        results.append({
            "question": q,
            "groundtruth": instance.get("groundtruth"),  # 添加 groundtruth 方便分析
            "answer_A": result_A["answer"],
            "score_A": float(result_A["score"]),
            "answer_B": result_B["answer"],
            "score_B": float(result_B["score"]),
            "answer_Conflict_Combined": result_Conflict["answer"],
            "score_Conflict_Combined": float(result_Conflict["score"])
        })

        if i < 2:
            print(f"\n[Q]: {q}")
            print(f"  Ans A ({result_A['score']:.4f}): {result_A['answer']}")
            print(f"  Ans B ({result_B['score']:.4f}): {result_B['answer']}")
            print(f"  Ans Combined ({result_Conflict['score']:.4f}): {result_Conflict['answer']}")

    return results


def run_factoid_conflict_qa(data: List[Dict[str, Any]], qa_func: Any):
    """处理事实冲突 (Factoid Conflicts) 任务。"""
    results = []
    print("\n" + "=" * 80)
    print(f"## 🅱️ 运行事实冲突问答任务 ({len(data)} 样本)")
    print("=" * 80)

    for i, instance in enumerate(tqdm(data, desc="Factoid Conflict")):
        q = instance["question"]

        context_Original = instance.get("original_evidence", "")
        if not context_Original:
            continue

        result_Original = qa_func(question=q, context=context_Original, top_k=1)

        context_Conflicting = instance.get("evidence", "")
        if not context_Conflicting:
            continue

        result_Conflicting = qa_func(question=q, context=context_Conflicting, top_k=1)

        # 记录结果，并确保分数是标准的 float 类型
        results.append({
            "question": q,
            "expected_answer": instance.get("answer"),
            "answer_Original": result_Original["answer"],
            "score_Original": float(result_Original["score"]),
            "answer_Conflicting": result_Conflicting["answer"],
            "score_Conflicting": float(result_Conflicting["score"])
        })

        if i < 2:
            print(f"\n[Q]: {q}")
            print(f"  Original Ans ({result_Original['score']:.4f}): {result_Original['answer']}")
            print(f"  Conflicting Ans ({result_Conflicting['score']:.4f}): {result_Conflicting['answer']}")
            print(f"  Expected: {instance.get('answer')}")

    return results


# --- 4. 文件保存函数 (保持不变) ---

def save_results_to_json(data: List[Dict[str, Any]], file_path: str):
    """将结果列表保存为 JSON 文件。"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 结果已成功保存到文件: {file_path} (共 {len(data)} 条记录)")
    except Exception as e:
        print(f"❌ 保存结果时发生错误: {e}")


# --- 5. 主程序执行 (更新保存逻辑) ---

if __name__ == "__main__":

    # 1. 加载数据
    answer_conflict_data = load_econ_data(ANSWER_CONFLICT_FILE, MAX_SAMPLES)
    factoid_conflict_data = load_econ_data(FACTOID_CONFLICT_FILE, MAX_SAMPLES)

    # 2. 运行问答任务和保存结果

    # 运行答案冲突任务
    if answer_conflict_data:
        answer_conflict_results = run_answer_conflict_qa(answer_conflict_data, qa_pipeline)

        # 保存答案冲突的结果
        if answer_conflict_results:
            save_results_to_json(answer_conflict_results, ANSWER_OUTPUT_FILE)

    # 运行事实冲突任务
    if factoid_conflict_data:
        factoid_conflict_results = run_factoid_conflict_qa(factoid_conflict_data, qa_pipeline)

        # 保存事实冲突的结果
        if factoid_conflict_results:
            save_results_to_json(factoid_conflict_results, FACTOID_OUTPUT_FILE)

    print("\n--- 任务完成 ---")
    print("您可以通过分析两个 JSON 文件中的答案和置信度，评估问答模型在不同冲突场景下的鲁棒性。")