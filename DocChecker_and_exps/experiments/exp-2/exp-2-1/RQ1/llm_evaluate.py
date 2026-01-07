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
import concurrent.futures  # 引入并发库
import threading  # 引入线程库
import time  # 引入时间库用于线程等待

# --- 配置 ---
MAX_WORKERS = 10  # 并发线程数
MAX_RETRIES = 3  # 最大重试次数
INPUT_FILE_PATH = "sampled_by_doc_type.json"  # 假设这是上一步生成的采样文件
TXT_OUTPUT_FILE = "report/LLM_doc_type_report.txt"  # 报告文件名

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 升级到 gpt-4o-mini
llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9
)

# 用于保护打印输出的锁 (在多线程环境下保证输出不混乱)
print_lock = threading.Lock()

# --- 1. LLM 提示和输出结构定义 ---

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


def call_llm_with_retry(evidence: str, reference: str, max_retries: int, unique_id: str) -> Optional[Dict[str, str]]:
    """调用 LLM 并处理重试和 JSON 解析错误。"""

    # 构造完整的用户提示（包含句子 A 和句子 B）
    user_prompt = LLM_PROMPT_TEMPLATE.format(evidence=evidence, reference_sentence=reference)

    # 系统消息为空字符串，因为所有指令都在 user_prompt 中
    system_instruction = "You are an accurate natural language inference model. Please output the results strictly according to the JSON format given by the user."

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                with print_lock:
                    print(f"[{threading.current_thread().name}] 重试 ({attempt}/{max_retries - 1}): ID {unique_id}")
                time.sleep(2 ** attempt)

            # LLM 调用
            response = llm.generate(
                user_message=user_prompt,
                system_message=system_instruction
            )

            # 兼容 OpenAIClient 返回文本或对象
            response_text = response.text if hasattr(response, 'text') else response

            # 清理和解析 JSON
            response_text = re.sub(r'```json\s*|```', '', response_text, flags=re.IGNORECASE).strip()

            result = json.loads(response_text)

            # 验证结构和内容
            if 'prediction_label' in result and 'prediction_contradiction_type' in result:
                return result
            else:
                raise ValueError(f"LLM返回的JSON缺少必要的键或格式不正确: {response_text}")

        except Exception as e:
            with print_lock:
                print(f"[{threading.current_thread().name}] ❌ LLM调用或解析失败 (尝试 {attempt + 1}/{max_retries})：ID {unique_id} - {e}")
            if attempt + 1 == max_retries:
                return None  # 达到最大重试次数
    return None


# --- 2. 线程工作函数 ---

def process_single_entry(unique_id: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    单个线程处理函数：负责提取数据、调用 LLM 并打包结果。
    """
    # ContraDOC 结构：'evidence' vs 'ref sentences' 列表中的第一句
    evidence = item.get('evidence', "")
    ref_sentences = item.get('ref sentences', [])
    reference = ref_sentences[0] if ref_sentences else ""

    if not evidence or not reference:
        return {
            'uid': unique_id,
            'gt_doc_type': item.get('doc_type', 'N/A').lower(),
            'gt_label': 'CONTRADICTION',
            'pred_label': 'ERROR',
            'pred_contradiction_type': 'N/A',
            'evidence': evidence,
            'reference': reference,
            'reason': 'Empty evidence or reference.'
        }

    # 提取 doc_type 作为分组字段
    gt_doc_type = item.get('doc_type', 'N/A')
    if isinstance(gt_doc_type, list): gt_doc_type = gt_doc_type[0]
    gt_doc_type = gt_doc_type.lower()

    gt_types = item.get('contra_type', ['N/A'])
    if isinstance(gt_types, list): gt_types = [t.strip() for t in gt_types if t.strip()]

    with print_lock:
        print(f"[{threading.current_thread().name}] 正在判断 ID: {unique_id} (Doc Type: {gt_doc_type.title()})...")

    llm_output = call_llm_with_retry(evidence, reference, MAX_RETRIES, unique_id)

    if llm_output:
        pred_label = llm_output['prediction_label'].upper()
        pred_type = llm_output['prediction_contradiction_type'].strip()
    else:
        pred_label = 'ERROR'
        pred_type = 'N/A'
        # call_llm_with_retry 内部已打印错误信息

    return {
        'uid': unique_id,
        'gt_doc_type': gt_doc_type,
        'gt_contradiction_type': gt_types,
        'gt_label': 'CONTRADICTION',
        'pred_label': pred_label,
        'pred_contradiction_type': pred_type,
        'evidence': evidence,
        'reference': reference
    }


# --- 3. LLM 判断执行函数 (多线程驱动) ---

def run_llm_contradiction_detection(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """使用 ThreadPoolExecutor 并发处理所有输入案例。"""

    pos_samples = data.get('pos', {})
    if not pos_samples:
        print("⚠️ 警告：输入文件中的 'pos' 字段为空或不存在。")
        return []

    total_samples = len(pos_samples)
    all_results = []
    futures_map = {}

    print(f"\n🧠 启动 LLM 矛盾判断 (多线程)，共 {total_samples} 个样本，最大线程数 {MAX_WORKERS}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for unique_id, item in pos_samples.items():
            # 提交任务，将 unique_id 和 item 传递给工作函数
            future = executor.submit(process_single_entry, unique_id, item)
            futures_map[future] = unique_id

        # 收集结果
        for i, future in enumerate(concurrent.futures.as_completed(futures_map)):
            unique_id = futures_map[future]

            try:
                result_entry = future.result()
                if result_entry:
                    all_results.append(result_entry)
            except Exception as e:
                with print_lock:
                    print(f"❌ 线程执行失败，ID {unique_id}：{e}")

            # 打印进度
            if (i + 1) % 50 == 0 or (i + 1) == total_samples:
                with print_lock:
                    print(f"    [进度] 已处理 {i + 1}/{total_samples} 个样本...")

    return all_results


# --- 4. 统计和报告函数 (保持不变) ---

def analyze_and_report(results: List[Dict[str, Any]]):
    """
    统计 LLM 性能，特别是按 doc_type 和 contra_type 的准确率。
    """

    # 1. 整体 NLI 准确率
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

    # 2. 按 Doc Type 统计 (核心修改)
    doc_type_stats = defaultdict(lambda: {'total': 0, 'correct_nli': 0, 'correct_type': 0, 'type_breakdown': Counter()})

    for res in results:
        if res['pred_label'] == 'ERROR':
            continue

        doc_type = res['gt_doc_type']  # 使用 doc_type 作为分组键

        doc_type_stats[doc_type]['total'] += 1

        # A. NLI 判断正确 (预测为 CONTRADICTION)
        is_nli_correct = res['pred_label'] == 'CONTRADICTION'
        if is_nli_correct:
            doc_type_stats[doc_type]['correct_nli'] += 1

        # B. 类型判断正确 (预测为 CONTRADICTION 且预测类型在 GT 类型列表中)
        is_type_correct = False
        if is_nli_correct and res['pred_contradiction_type'].lower() in [t.lower() for t in res['gt_contradiction_type']]:
            doc_type_stats[doc_type]['correct_type'] += 1
            is_type_correct = True

        # C. 类型细分统计 (统计 LLM 预测的类型分布和 LLM 预测正确率)
        if is_nli_correct:
            doc_type_stats[doc_type]['type_breakdown'][res['pred_contradiction_type'].lower()] += 1

    # 3. 格式化报告
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("        🤖 LLM 驱动的 ContraDOC 矛盾分析报告 (GPT-4o mini) - 多线程")
    report_lines.append("=" * 70)
    report_lines.append(f"总样本数 (来自 pos): {len(results)}")
    report_lines.append(f"成功分析样本数: {total_analyzed}")
    report_lines.append(f"LLM 错误/解析失败样本数: {total_error}\n")

    # A. 整体 NLI 指标
    report_lines.append("--- 1. 整体 NLI 性能 (GT: CONTRADICTION) ---")
    report_lines.append(f"准确率/召回率 (LLM 预测 CONTRADICTION): {overall_metrics['Accuracy']:.2%}")
    report_lines.append(f"F1 Score: {overall_metrics['F1']:.2%}\n")

    # B. 按 Doc Type 统计
    report_lines.append("--- 2. 按 Doc Type 划分的详细性能 ---")

    # 构建表格
    headers = ["Doc Type", "样本数", "NLI准确率(%)", "类型匹配率(%)"]
    table_lines = [" | ".join(headers)]
    table_lines.append("|" + "---|" * len(headers))

    for doc_type in sorted(doc_type_stats.keys()):
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

    # C. 按 Doc Type 的类型预测细分
    report_lines.append("--- 3. LLM 预测的矛盾类型分布 (按 Doc Type 划分) ---")
    report_lines.append("(仅统计LLM预测为 CONTRADICTION 的样本)")

    for doc_type in sorted(doc_type_stats.keys()):
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


# --- 4. 主执行块 ---

if __name__ == '__main__':

    # 1. 加载采样数据
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"❌ 错误：找不到输入文件 '{INPUT_FILE_PATH}'。请先运行采样脚本生成该文件。")
        exit()

    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            sampled_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取或解析文件 '{INPUT_FILE_PATH}' 失败: {e}")
        exit()

    # 2. 执行 LLM 判断
    llm_results = run_llm_contradiction_detection(sampled_data)

    if llm_results:
        # 3. 统计和报告
        analyze_and_report(llm_results)

    print("\n--- LLM 分析任务完成 ---")