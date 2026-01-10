from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
import difflib
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import concurrent.futures
import threading

# --- 配置 ---
MAX_RETRIES = 3  # 最大重试次数
# ⚠️ 文件名已根据用户要求更新为 dense_text.json
INPUT_FILE_PATH = "dense_text.json"
# TXT_OUTPUT_FILE = "report/ContraDoc_report.txt"
TXT_OUTPUT_FILE = "report/ContraDoc_report-gpt-4o.txt"
MAX_WORKERS = 8  # 增加并发线程数以加速 I/O 任务

# 加载环境变量
load_dotenv()

# 确保环境变量设置
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 检查 API Key 是否存在
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 环境变量未设置。")

llm = OpenAIClient(
        model="gpt-4o",
        temperature=0.1,
        top_p=0.9
)

# 用于保护打印输出的锁 (可选，但推荐用于清晰的并发输出)
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
    user_prompt = LLM_PROMPT_TEMPLATE.format(evidence=evidence, reference_sentence=reference)
    system_instruction = "You are an accurate NLI model. Output ONLY JSON."

    for attempt in range(max_retries):
        try:
            response = llm.generate(
                user_message=user_prompt,
                system_message=system_instruction
            )

            # --- 关键修复：增强提取逻辑 ---
            if not response:
                raise ValueError("API 返回了空响应")

            response_text = getattr(response, 'text', str(response))

            # 使用正则提取第一个 { ... } 之间的内容，防止模型输出前后带废话
            json_match = re.search(r'\{.*\}', response_text.replace('\n', ' '), re.DOTALL)
            if not json_match:
                raise ValueError(f"无法在响应中找到 JSON 结构: {response_text[:100]}...")

            cleaned_json_str = json_match.group(0)
            result = json.loads(cleaned_json_str)

            # 验证必要字段
            if 'prediction_label' in result:
                # 规范化 key 名（防止模型返回小写或不规范 key）
                final_result = {
                    "prediction_label": str(result.get("prediction_label", "NEUTRAL")).upper(),
                    "prediction_contradiction_type": str(result.get("prediction_contradiction_type", "N/A"))
                }
                return final_result
            else:
                raise ValueError("JSON 缺少 prediction_label 字段")

        except Exception as e:
            with print_lock:
                print(f"    ❌ ID {uid}: 尝试 {attempt + 1}/{max_retries} 失败：{e}")
            if attempt + 1 == max_retries:
                return None
            # 在重试前稍作停顿，避免触发中转站 QPS 限制
            import time
            time.sleep(1)
    return None


# --- 2. 数据转换函数 ---
def flatten_nested_data(nested_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将嵌套的文档/矛盾对结构展平为单一的句子对列表。

    输入: [ {..., "scope": "global", "contradiction_pairs": [ { "text": "A [SEP] B", ... }, ... ] }, ... ]
    输出: [ { "uid": "...", "gt_scope": "global", "gt_contradiction_type": ["Factual"], "text": "A [SEP] B", ... }, ... ]
    """
    flat_list = []

    for doc_entry in nested_data:
        source_id = doc_entry.get("source_id", "unknown_doc")

        # 优化点：在展平时就标准化 scope 字段
        scope_raw = doc_entry.get("scope", "N/A")
        if isinstance(scope_raw, list):
            # 如果是列表，取第一个元素
            scope = scope_raw[0].lower() if scope_raw and isinstance(scope_raw[0], str) else 'n/a'
        elif isinstance(scope_raw, str):
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

            # 构建扁平化条目，供 run_llm_contradiction_detection 使用
            flat_list.append({
                'uid': uid,
                'gt_scope': scope,
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
    增加了超时保护和全局异常捕获，确保程序不会因线程异常而崩溃。
    """
    results = []
    total_samples = len(data_list)
    tasks = []  # 存储 (Future, item_data, index)

    print(f"\n🧠 启动 LLM 矛盾判断，共 {total_samples} 个句子对样本，并行线程数: {MAX_WORKERS}...")

    # 使用 ThreadPoolExecutor 实现并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, item in enumerate(data_list):
            evidence = item.get('evidence', '')
            reference = item.get('reference', '')
            uid = item.get('uid', f"sample_{i + 1}")

            if not evidence or not reference:
                with print_lock:
                    print(f"    ⚠️ ID {uid}: 句子缺失（Evidence 或 Reference 为空），跳过。")
                continue

            # 提交任务
            future = executor.submit(call_llm_with_retry, evidence, reference, MAX_RETRIES, uid)
            tasks.append((future, item, i))

        # 收集结果
        for j, (future, item, i) in enumerate(tasks):
            uid = item.get('uid', f"sample_{i + 1}")

            try:
                # 增加 60 秒超时，防止某个请求挂起导致整个程序堵塞
                llm_output = future.result(timeout=60)
            except concurrent.futures.TimeoutError:
                with print_lock:
                    print(f"    🚨 ID {uid}: 任务超时 (60s)，标记为 ERROR。")
                llm_output = None
            except Exception as e:
                with print_lock:
                    print(f"    🚨 ID {uid}: 线程内部发生未捕获异常: {e}")
                llm_output = None

            # 打印进度 (每 50 个样本或最后一个)
            if (j + 1) % 50 == 0 or (j + 1) == total_samples:
                with print_lock:
                    print(f"    [进度] 已处理 {j + 1}/{total_samples} ({(j + 1) / total_samples:.1%})")

            # 提取原有的 Ground Truth 信息用于对比
            gt_scope_raw = item.get('gt_scope', 'n/a')
            if isinstance(gt_scope_raw, list):
                gt_scope = gt_scope_raw[0].lower() if gt_scope_raw and isinstance(gt_scope_raw[0], str) else 'n/a'
            else:
                gt_scope = str(gt_scope_raw).lower()

            gt_types = item.get('gt_contradiction_type', ['N/A'])

            # 处理 LLM 的预测结果
            if llm_output and isinstance(llm_output, dict):
                pred_label = str(llm_output.get('prediction_label', 'NEUTRAL')).upper()
                pred_type = str(llm_output.get('prediction_contradiction_type', 'N/A'))
            else:
                # 如果失败，标记为 ERROR 以便在分析报告中体现
                pred_label = 'ERROR'
                pred_type = 'N/A'

            results.append({
                'uid': uid,
                'gt_scope': gt_scope,
                'gt_contradiction_type': gt_types,
                'gt_label': 'CONTRADICTION',
                'pred_label': pred_label,
                'pred_contradiction_type': pred_type,
                'evidence': item.get('evidence', ''),
                'reference': item.get('reference', '')
            })

    print(f"\n✅ LLM 判断执行完毕。成功回收 {len(results)} 条结果。")
    return results

# --- 4. 统计和报告函数 (保持不变) ---

def analyze_and_report(results: List[Dict[str, Any]]):
    """
    统计 LLM 性能，特别是按 scope 和 contra_type 的准确率。
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

    # 2. 按 Scope 统计
    scope_stats = defaultdict(lambda: {'total': 0, 'correct_nli': 0, 'correct_type': 0, 'type_breakdown': Counter()})

    for res in results:
        if res['pred_label'] == 'ERROR':
            continue

        scope = res['gt_scope']  # 使用展平后的 gt_scope

        scope_stats[scope]['total'] += 1

        # A. NLI 判断正确 (预测为 CONTRADICTION)
        is_nli_correct = res['pred_label'] == 'CONTRADICTION'
        if is_nli_correct:
            scope_stats[scope]['correct_nli'] += 1

        # B. 类型判断正确 (预测为 CONTRADICTION 且预测类型在 GT 类型列表中)
        gt_type_list = [t.lower() for t in res['gt_contradiction_type'] if t.strip()]

        if is_nli_correct and res['pred_contradiction_type'].lower() in gt_type_list:
            scope_stats[scope]['correct_type'] += 1

        # C. 类型细分统计 (统计 LLM 预测的类型分布)
        if is_nli_correct:
            scope_stats[scope]['type_breakdown'][res['pred_contradiction_type'].lower()] += 1

    # 3. 格式化报告
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("        🤖 LLM 驱动的 ContraDOC 矛盾分析报告 (GPT-4o mini)")
    report_lines.append("=" * 70)
    report_lines.append(f"总句子对样本数: {len(results)}")
    report_lines.append(f"成功分析样本数: {total_analyzed}")
    report_lines.append(f"LLM 错误/解析失败样本数: {total_error}\n")

    # A. 整体 NLI 指标
    report_lines.append("--- 1. 整体 NLI 性能 (GT: CONTRADICTION) ---")
    report_lines.append(f"准确率/召回率 (LLM 预测 CONTRADICTION): {overall_metrics['Accuracy']:.2%}")
    report_lines.append(f"F1 Score: {overall_metrics['F1']:.2%}\n")

    # B. 按 Scope 统计
    report_lines.append("--- 2. 按 Scope 划分的详细性能 ---")

    # 构建表格
    headers = ["Scope", "样本数", "NLI准确率(%)", "类型匹配率(%)"]
    table_lines = [" | ".join(headers)]
    table_lines.append("|" + "---|" * len(headers))

    # 排序并生成表格行
    sorted_scopes = sorted(scope_stats.keys())
    if 'n/a' in sorted_scopes:
        # 将 N/A 移到最后
        sorted_scopes.remove('n/a')
        sorted_scopes.append('n/a')

    for scope in sorted_scopes:
        stats = scope_stats[scope]
        total = stats['total']
        nli_acc = stats['correct_nli'] / total if total > 0 else 0
        type_match_rate = stats['correct_type'] / total if total > 0 else 0

        row = [
            scope.title(),
            str(total),
            f"{nli_acc:.2%}",
            f"{type_match_rate:.2%}"
        ]
        table_lines.append(" | ".join(row))

    report_lines.extend(table_lines)
    report_lines.append("\n" + "=" * 70)

    # C. 按 Scope 的类型预测细分
    report_lines.append("--- 3. LLM 预测的矛盾类型分布 (按 Scope 划分) ---")
    report_lines.append("(仅统计LLM预测为 CONTRADICTION 的样本)")

    for scope in sorted_scopes:
        report_lines.append(f"\n>> Scope: {scope.title()}")
        breakdown = scope_stats[scope]['type_breakdown']
        total_scope_pred = sum(breakdown.values())
        if total_scope_pred == 0:
            report_lines.append("   - 无预测为 CONTRADICTION 的样本。")
            continue

        for pred_type, count in sorted(breakdown.items(), key=lambda item: item[1], reverse=True):
            report_lines.append(f"   - {pred_type.title()}: {count} ({count / total_scope_pred:.2%})")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("说明:")
    report_lines.append("- NLI 准确率: LLM 预测为 CONTRADICTION 的样本占总样本数的比例 (因 GT 均为 CONTRADICTION，即召回率)。")
    report_lines.append("- 类型匹配率: LLM 预测为 CONTRADICTION 且其预测类型包含在 GT 矛盾类型列表中的样本比例。")
    report_lines.append("- LLM 预测类型为 N/A 的样本未计入类型细分统计。")
    report_lines.append("=" * 70)

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
            # 兼容旧的字典格式，但重点处理列表格式
            if isinstance(data_list, dict) and 'pos' in data_list:
                # 如果是旧格式，将其转换为列表
                data_list = list(data_list['pos'].values())
            elif not isinstance(data_list, list):
                print(f"❌ 错误：输入文件 '{INPUT_FILE_PATH}' 格式不正确。预期为 JSON 列表或包含 'pos' 键的字典。")
                exit()
    except Exception as e:
        print(f"❌ 读取或解析文件 '{INPUT_FILE_PATH}' 失败: {e}")
        exit()

    # 2. 🎯 展平数据：将嵌套的文档/矛盾对结构转换为扁平的句子对列表
    # 这一步将新输入格式转换为原始代码可以处理的扁平结构，并保留 scope 和 type 信息。
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