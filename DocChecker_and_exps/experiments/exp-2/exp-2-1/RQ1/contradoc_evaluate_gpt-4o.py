"""
NLI准确率(%) | 类型匹配率(%),这两个指标都是用于衡量您的语言模型（LLM，即 GPT-4o mini）在处理 ContraDOC 矛盾样本时的性能，
并且都是在 Ground Truth (GT) 标签为 CONTRADICTION 的前提下计算的（因为样本来自于 'pos' 集合）。

--- 报告指标详细说明 ---

本报告中的所有样本均来自 ContraDOC 数据集的 'pos' (Positive) 集合，即 Ground Truth (GT) 标签均为 CONTRADICTION。

===============================================================
1. NLI 准确率 (%)
===============================================================
**对应指标:** NLI 准确率 (%)
**定义:** 召回率 (Recall)。它衡量了 LLM 模型正确识别出句子对中存在矛盾的能力。

**计算公式:**
NLI 准确率 = (LLM 预测为 CONTRADICTION 的样本数) / (总样本数)

**含义:**
在所有确实存在矛盾的句子对中，您的 LLM 模型成功地将其标记为 "CONTRADICTION" 的比例。这个指标主要关注模型在二分类任务（矛盾 vs. 非矛盾）中的敏感度。

**示例:**
如果 News 类型有 109 个样本，模型正确预测出 68 个矛盾，则 NLI 准确率为 68/109 ≈ 62.39%。
**观察:**
在 Doc Type 统计中，NLI 准确率体现了模型在不同类型文档（Story, News, Wiki）背景下对矛盾的整体敏感性。

===============================================================
2. 类型匹配率 (%)
===============================================================
**对应指标:** 类型匹配率 (%)
**定义:** 准确识别矛盾并正确分类的能力。这是对模型细粒度推理能力的要求。

**计算公式:**
类型匹配率 = (LLM 预测为 CONTRADICTION 且 预测类型在 GT 类型列表中的样本数) / (总样本数)

**含义:**
在所有确实存在矛盾的句子对中，您的模型必须同时满足两个条件：
1. 正确预测标签为 "CONTRADICTION"。
2. 预测的矛盾类型（如 "Numeric", "Temporal" 等）必须包含在人工标注的 GT 矛盾类型列表中。

**观察:**
类型匹配率通常远低于 NLI 准确率。这表明 LLM 识别出矛盾（NLI）相对容易，但要对矛盾进行精确的细粒度分类则非常困难。

---
"""

import json
import random
import os
import re
import difflib
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import concurrent.futures
import threading
import time

# --- 配置 ---
MAX_RETRIES = 3  # 最大重试次数
# ⚠️ 文件名已根据用户要求更新为 dense_text.json
INPUT_FILE_PATH = "dense_text.json"
# 更改报告文件名以反映按 doc_type 评估
TXT_OUTPUT_FILE = "report/ContraDoc_doc_type_report_gpt-4o.txt"
MAX_WORKERS = 10  # 并发线程数

# 加载环境变量
load_dotenv()

# 确保环境变量设置
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 检查 API Key 是否存在
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 环境变量未设置。")

# 升级到 gpt-4o-mini
llm = OpenAIClient(
    model="gpt-4o",
    temperature=0.1,
    top_p=0.9
)

# 用于保护打印输出的锁
print_lock = threading.Lock()

# --- 1. LLM 提示和输出结构定义 (保持不变) ---
LLM_PROMPT_TEMPLATE = """
任务：对给定的两个句子进行自然语言推理 (NLI) 判断，确定它们之间是否存在矛盾（Contradiction），并识别矛盾的类型。

句子 A (Ground Truth Evidence): "{evidence}"
句子 B (Ground Truth Reference): "{reference_sentence}"

请严格按照以下 JSON 格式返回结果。只返回 JSON 对象，不要包含任何额外的文本或解释。

要求返回的 JSON 格式:
{{
    "prediction_label": "CONTRADICTION" | "ENTAILMENT" | "NEUTRAL",
    "prediction_contradiction_type": "Content" | "Numeric" | "Causal" | "Negation" | "Perspective/View/Opinion" | "Emotion/Mood/Feeling" | "Temporal" | "N/A"
}}

如果预测结果不是 CONTRADICTION，则 prediction_contradiction_type 必须是 "N/A"。
如果预测结果是 CONTRADICTION，请从提供的列表中选择最合适的矛盾类型。
"""

LLM_TYPES_LIST = [
    "Content", "Numeric", "Causal", "Negation",
    "Perspective/View/Opinion", "Emotion/Mood/Feeling",
    "Temporal", "N/A"
]


def call_llm_with_retry(evidence: str, reference: str, max_retries: int, uid: str) -> Optional[Dict[str, str]]:
    """调用 LLM 并处理重试和 JSON 解析错误。"""

    user_prompt = LLM_PROMPT_TEMPLATE.format(evidence=evidence, reference_sentence=reference)
    system_instruction = "You are an accurate natural language inference model. Please output the results strictly according to the JSON format given in the user message."

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                with print_lock:
                    print(f"[{threading.current_thread().name}] 重试 ({attempt}/{max_retries - 1}): ID {uid}")
                time.sleep(2 ** attempt)

            # 🚀 LLM 调用行
            response = llm.generate(
                user_message=user_prompt,
                system_message=system_instruction
            )

            response_text = getattr(response, 'text', response)
            response_text = re.sub(r'```json\s*|```', '', response_text, flags=re.IGNORECASE).strip()

            result = json.loads(response_text)

            if 'prediction_label' in result and 'prediction_contradiction_type' in result:
                pred_type = result['prediction_contradiction_type'].strip()
                if pred_type not in LLM_TYPES_LIST:
                    with print_lock:
                        print(f"    ⚠️ ID {uid}: LLM返回了未知的矛盾类型: {pred_type}，尝试转换为 N/A。")
                    result['prediction_contradiction_type'] = 'N/A'
                return result
            else:
                raise ValueError(f"LLM返回的JSON缺少必要的键或格式不正确: {response_text}")

        except Exception as e:
            with print_lock:
                # 使用 print_lock 确保并发输出清晰
                print(f"    ❌ ID {uid}: LLM调用或解析失败 (尝试 {attempt + 1}/{max_retries})：{e}")
            if attempt + 1 == max_retries:
                return None
    return None


# --- 2. 数据转换函数 ---
def flatten_nested_data(nested_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将嵌套的文档/矛盾对结构展平为单一的句子对列表，并提取 doc_type 作为分组字段。
    """
    flat_list = []

    for doc_entry in nested_data:
        source_id = doc_entry.get("source_id", "unknown_doc")

        # 🎯 提取 doc_type 字段
        doc_type_raw = doc_entry.get("doc_type", "N/A")
        if isinstance(doc_type_raw, list):
            doc_type = doc_type_raw[0].lower() if doc_type_raw and isinstance(doc_type_raw[0], str) else 'n/a'
        elif isinstance(doc_type_raw, str):
            doc_type = doc_type_raw.lower()
        else:
            doc_type = 'n/a'

        # 原始的 scope 字段也保留，但用于报告分组的是 doc_type
        scope_raw = doc_entry.get("scope", "N/A")
        if isinstance(scope_raw, list):
            scope = scope_raw[0].lower() if scope_raw and isinstance(scope_raw[0], str) else 'n/a'
        elif isinstance(scope_raw, str):
            # ⬇️ 修正：将 scope_type 替换为 scope_raw
            scope = scope_raw.lower()
        else:
            scope = 'n/a'


        pairs = doc_entry.get("contradiction_pairs", [])

        # 如果是成功的文档，但没有提取出矛盾对，则跳过
        if not pairs and doc_entry.get("status") == "Success":
            continue

        for i, pair in enumerate(pairs):
            # 构造唯一的ID，例如 document_id_pair_index
            uid = f"{source_id}_{i}"

            # 从 pair 中提取信息
            pair_text = pair.get("text", "[SEP]")
            contra_type = pair.get("contradiction_type", "N/A")

            # 提取句子 A 和 B
            text_pair = pair_text.split('[SEP]')
            evidence = text_pair[0].strip() if len(text_pair) > 0 else ""
            reference = text_pair[1].strip() if len(text_pair) > 1 else ""

            # 构建扁平化条目
            flat_list.append({
                'uid': uid,
                'gt_doc_type': doc_type, # 🎯 展平结构中现在使用 gt_doc_type
                'gt_scope': scope,       # scope 字段作为辅助信息保留
                'gt_contradiction_type': [contra_type],  # 保持列表结构
                'gt_label': 'CONTRADICTION',  # 假设所有提取的对都是矛盾
                'text': pair_text,
                'evidence': evidence,
                'reference': reference
            })

    return flat_list


# --- 3. LLM 判断执行函数 (多线程实现) ---

def run_llm_contradiction_detection(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    遍历扁平化的句子对列表，使用多线程并发调用 LLM 进行矛盾判断。
    """
    results = []
    total_samples = len(data_list)
    tasks = []  # 存储 Future 对象

    print(f"\n🧠 启动 LLM 矛盾判断，共 {total_samples} 个句子对样本，使用 {MAX_WORKERS} 个线程...")

    # 使用 ThreadPoolExecutor 实现并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, item in enumerate(data_list):

            # 提取扁平化数据中的句子
            evidence = item.get('evidence', '')
            reference = item.get('reference', '')
            uid = item.get('uid', f"sample_{i + 1}")

            if not evidence or not reference:
                with print_lock:
                    print(f"    ❌ 样本 {uid} 句子缺失，跳过。")
                continue

            # 提交任务到线程池
            future = executor.submit(call_llm_with_retry, evidence, reference, MAX_RETRIES, uid)

            # 将 future 和 item 的 GT 信息一起保存
            tasks.append((future, item, i))

        # 收集结果
        for j, (future, item, i) in enumerate(tasks):
            uid = item.get('uid', f"sample_{i + 1}")

            try:
                llm_output = future.result()  # 阻塞直到该线程任务完成
            except Exception as e:
                 with print_lock:
                    print(f"    ❌ 线程 {future} 意外失败: {e}")
                    llm_output = None


            # 报告进度
            if (j + 1) % 50 == 0 or (j + 1) == total_samples:
                with print_lock:
                    print(f"    [进度] 已处理 {j + 1}/{total_samples} 个句子对样本...")

            # 提取原始 GT 信息
            gt_doc_type = item.get('gt_doc_type', 'n/a') # 🎯 使用 doc_type
            gt_types = item.get('gt_contradiction_type', ['N/A'])
            evidence = item.get('evidence', '')
            reference = item.get('reference', '')

            if llm_output:
                pred_label = llm_output['prediction_label'].upper()
                pred_type = llm_output['prediction_contradiction_type'].strip()
            else:
                pred_label = 'ERROR'
                pred_type = 'N/A'
                # 错误信息已在 call_llm_with_retry 中打印

            results.append({
                'uid': uid,
                'gt_doc_type': gt_doc_type, # 🎯 输出结果中保存 doc_type
                'gt_contradiction_type': gt_types,
                'gt_label': 'CONTRADICTION',
                'pred_label': pred_label,
                'pred_contradiction_type': pred_type,
                'evidence': evidence,
                'reference': reference
            })

    print("\n✅ LLM 判断执行完毕。")
    return results


# --- 4. 统计和报告函数 (修改为 Doc Type) ---

def analyze_and_report(results: List[Dict[str, Any]]):
    """
    统计 LLM 性能，特别是按 doc_type 和 contra_type 的准确率。
    """

    # 1. 整体 NLI 准确率 (LLM 预测 CONTRADICTION vs. GT CONTRADICTION)
    y_true_nli = np.array([1 for res in results if res['pred_label'] != 'ERROR'])
    y_pred_nli = np.array([1 if res['pred_label'] == 'CONTRADICTION' else 0
                           for res in results if res['pred_label'] != 'ERROR'])

    total_analyzed = len(y_true_nli)
    total_error = len(results) - total_analyzed

    overall_metrics = {
        'Accuracy': accuracy_score(y_true_nli, y_pred_nli) if total_analyzed > 0 else 0,
        'Precision': precision_score(y_true_nli, y_pred_nli, zero_division=0) if total_analyzed > 0 else 0,
        'Recall': recall_score(y_true_nli, y_pred_nli, zero_division=0) if total_analyzed > 0 else 0,
        'F1': f1_score(y_true_nli, y_pred_nli, zero_division=0) if total_analyzed > 0 else 0
    }

    # 2. 按 Doc Type 统计 🎯
    doc_type_stats = defaultdict(lambda: {'total': 0, 'correct_nli': 0, 'correct_type': 0, 'type_breakdown': Counter()})

    for res in results:
        if res['pred_label'] == 'ERROR':
            continue

        doc_type = res['gt_doc_type']  # 🎯 使用 gt_doc_type 进行分组

        doc_type_stats[doc_type]['total'] += 1

        # A. NLI 判断正确 (预测为 CONTRADICTION)
        is_nli_correct = res['pred_label'] == 'CONTRADICTION'
        if is_nli_correct:
            doc_type_stats[doc_type]['correct_nli'] += 1

        # B. 类型判断正确 (预测为 CONTRADICTION 且预测类型在 GT 类型列表中)
        gt_type_list = [t.lower() for t in res['gt_contradiction_type'] if t.strip()]

        if is_nli_correct and res['pred_contradiction_type'].lower() in gt_type_list:
            doc_type_stats[doc_type]['correct_type'] += 1

        # C. 类型细分统计 (统计 LLM 预测的类型分布)
        if is_nli_correct:
            doc_type_stats[doc_type]['type_breakdown'][res['pred_contradiction_type'].lower()] += 1

    # 3. 格式化报告
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("        🤖 LLM 驱动的 ContraDOC 矛盾分析报告 (GPT-4o mini) - 按 Doc Type 评估")
    report_lines.append("=" * 70)
    report_lines.append(f"总句子对样本数: {len(results)}")
    report_lines.append(f"成功分析样本数: {total_analyzed}")
    report_lines.append(f"LLM 错误/解析失败样本数: {total_error}\n")

    # A. 整体 NLI 指标
    report_lines.append("--- 1. 整体 NLI 性能 (GT: CONTRADICTION) ---")
    report_lines.append(f"准确率/召回率 (LLM 预测 CONTRADICTION): {overall_metrics['Accuracy']:.2%}")
    report_lines.append(f"F1 Score: {overall_metrics['F1']:.2%}\n")

    # B. 按 Doc Type 统计 🎯
    report_lines.append("--- 2. 按 Doc Type 划分的详细性能 ---")

    # 构建表格
    headers = ["Doc Type", "样本数", "NLI准确率(%)", "类型匹配率(%)"]
    table_lines = [" | ".join(headers)]
    table_lines.append("|" + "---|" * len(headers))

    # 排序并生成表格行
    sorted_doc_types = sorted(doc_type_stats.keys())
    if 'n/a' in sorted_doc_types:
        # 将 N/A 移到最后
        sorted_doc_types.remove('n/a')
        sorted_doc_types.append('n/a')

    for doc_type in sorted_doc_types:
        stats = doc_type_stats[doc_type]
        total = stats['total']
        nli_acc = stats['correct_nli'] / total if total > 0 else 0
        type_match_rate = stats['correct_type'] / total if total > 0 else 0

        row = [
            doc_type.title(),
            str(total),
            f"{nli_acc:.2%}",
            f"{type_match_rate:.2%}"
        ]
        table_lines.append(" | ".join(row))

    report_lines.extend(table_lines)
    report_lines.append("\n" + "=" * 70)

    # C. 按 Doc Type 的类型预测细分 🎯
    report_lines.append("--- 3. LLM 预测的矛盾类型分布 (按 Doc Type 划分) ---")
    report_lines.append("(仅统计LLM预测为 CONTRADICTION 的样本)")

    for doc_type in sorted_doc_types:
        report_lines.append(f"\n>> Doc Type: {doc_type.title()}")
        breakdown = doc_type_stats[doc_type]['type_breakdown']
        total_type_pred = sum(breakdown.values())
        if total_type_pred == 0:
            report_lines.append("   - 无预测为 CONTRADICTION 的样本。")
            continue

        for pred_type, count in sorted(breakdown.items(), key=lambda item: item[1], reverse=True):
            report_lines.append(f"   - {pred_type.title()}: {count} ({count / total_type_pred:.2%})")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("说明:")
    report_lines.append("- NLI 准确率: LLM 预测为 CONTRADICTION 的样本占总样本数的比例 (因 GT 均为 CONTRADICTION，即召回率)。")
    report_lines.append("- 类型匹配率: LLM 预测为 CONTRADICTION 且其预测类型包含在 GT 矛盾类型列表中的样本比例。")
    report_lines.append("- LLM 预测类型为 N/A 的样本未计入类型细分统计。")
    report_lines.append("=" * 70)

    # 确保 report 目录存在
    report_dir = os.path.dirname(TXT_OUTPUT_FILE)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # 写入文件
    try:
        with open(TXT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"\n🎉 报告已成功保存到文件: {TXT_OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 保存报告时发生错误: {e}")


# --- 5. 主执行块 ---

if __name__ == '__main__':

    # 1. 加载数据
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"❌ 错误：找不到输入文件 '{INPUT_FILE_PATH}'。请检查文件路径。")
        exit()

    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            # 兼容旧的字典格式，但重点处理列表格式 (来自 dense_text.json 的预期格式)
            if isinstance(data_list, dict) and 'pos' in data_list:
                data_list = list(data_list['pos'].values())
            elif not isinstance(data_list, list):
                print(f"❌ 错误：输入文件 '{INPUT_FILE_PATH}' 格式不正确。预期为 JSON 列表或包含 'pos' 键的字典。")
                exit()
    except Exception as e:
        print(f"❌ 读取或解析文件 '{INPUT_FILE_PATH}' 失败: {e}")
        exit()

    # 2. 🎯 展平数据：将嵌套的文档/矛盾对结构转换为扁平的句子对列表
    flat_data = flatten_nested_data(data_list)

    if not flat_data:
        print("⚠️ 警告: 展平后的矛盾句子对列表为空。没有需要分析的样本。")
        exit()

    # 3. 执行 LLM 判断 (多线程)
    llm_results = run_llm_contradiction_detection(flat_data)

    if llm_results:
        # 4. 统计和报告
        analyze_and_report(llm_results)

    print("\n--- LLM 分析任务完成 ---")