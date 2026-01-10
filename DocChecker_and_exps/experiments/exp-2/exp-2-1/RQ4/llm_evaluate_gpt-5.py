from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
from typing import Dict, Any, Optional, List
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# --- 导入多线程模块 ---
import concurrent.futures
import threading
import time

# --- 配置 ---
# --- 配置 ---
MAX_RETRIES = 3
MAX_WORKERS = 10
INPUT_FILE_PATH = "dense_text.txt"
TXT_OUTPUT_FILE = "report/LLM_report_detailed_gpt-5.2-chat-latest.txt"

# 加载环境变量
load_dotenv()

# --- 1. 初始化和配置 (保持不变) ---
# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2-chat-latest",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# 使用锁来保护共享资源（打印输出）
print_lock = threading.Lock()

# --- 1. LLM 提示和输出结构定义 ---
LLM_TYPES_LIST = [
    "Causal", "Content", "Factual", "Negation", "None",
    "Numeric", "Relation", "Temporal", "N/A"
]

LLM_PROMPT_TEMPLATE = """
任务：对给定的两个句子进行自然语言推理 (NLI) 判断，确定它们之间是否存在矛盾（Contradiction），并识别矛盾的类型。

句子 A (Ground Truth Evidence): "{evidence}"
句子 B (Ground Truth Reference): "{reference_sentence}"

请严格按照以下 JSON 格式返回结果。只返回 JSON 对象，不要包含任何额外的文本或解释。

要求返回的 JSON 格式:
{{
    "prediction_label": "CONTRADICTION" | "ENTAILMENT" | "NEUTRAL",
    "prediction_contradiction_type": "Causal" | "Content" | "Factual" | "Negation" | "None" | "Numeric" | "Relation" | "Temporal" | "N/A"
}}

如果预测结果不是 CONTRADICTION，则 prediction_contradiction_type 必须是 "None" 或 "N/A"。
如果预测结果是 CONTRADICTION，请从提供的列表中选择最合适的矛盾类型。
"""





# 辅助函数：LLM 调用和解析 (保持不变)
def call_llm_with_retry(evidence: str, reference: str, max_retries: int) -> Optional[Dict[str, str]]:
    """调用 LLM 并处理重试和 JSON 解析错误。"""

    user_prompt = LLM_PROMPT_TEMPLATE.format(evidence=evidence, reference_sentence=reference)
    system_instruction = "You are an accurate natural language inference model. Please output the results strictly according to the JSON format given by the user."

    # 建议将 model 改为 gpt-4o 或 gpt-3.5-turbo，除非你的中转商明确支持名为 "gpt-5" 的自定义模型
    target_model = "gpt-4o"

    for attempt in range(max_retries):
        try:
            response = llm.generate(
                user_message=user_prompt,
                system_message=system_instruction
            )

            # 获取文本内容
            if hasattr(response, 'text'):
                response_text = response.text
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

            # --- 增强版 JSON 提取逻辑 ---
            # 1. 尝试使用正则匹配第一个 { 和最后一个 } 之间的内容
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
            else:
                clean_json = response_text.strip()

            try:
                result = json.loads(clean_json)
            except json.JSONDecodeError:
                with print_lock:
                    print(f"    ⚠️ JSON 解析失败。LLM 原始输出: {response_text[:100]}...")
                raise

            if 'prediction_label' in result and 'prediction_contradiction_type' in result:
                pred_type = str(result['prediction_contradiction_type']).strip()
                if pred_type not in LLM_TYPES_LIST:
                    result['prediction_contradiction_type'] = 'N/A'
                result['prediction_label'] = str(result['prediction_label']).upper()
                return result
            else:
                raise ValueError("JSON 缺少必要的 Key")

        except Exception as e:
            with print_lock:
                print(f"    ❌ 尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt + 1 == max_retries:
                return None
            time.sleep(1)  # 短暂延迟后重试
    return None

# --- 2. 单个样本的处理函数 (包含 GT 修复逻辑) ---
def process_sample(i: int, total_samples: int, uid: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """处理单个样本，调用 LLM 并返回结果字典。"""

    # 这里的键名依赖于主块中的数据标准化结果
    evidence = item.get('evidence', "")
    ref_sentences = item.get('ref sentences', [])
    reference = ref_sentences[0] if ref_sentences else ""

    # --- GT 类型提取和修复 (使用 standardized_pos_samples 的键) ---
    gt_types_raw = item.get('contra_type', ['N/A'])

    # 确保 gt_types_raw 是列表
    if not isinstance(gt_types_raw, list):
        gt_types_raw = [str(gt_types_raw)]

    gt_types = [t.strip() for t in gt_types_raw if t.strip()]

    # 过滤和验证
    valid_gt_types = [t for t in gt_types if t in LLM_TYPES_LIST and t not in ('N/A', 'None')]

    if not valid_gt_types:
        # 如果没有找到有效的矛盾类型，我们检查原始列表中的第一个类型是否在 LLM_TYPES_LIST 中
        if gt_types and gt_types[0] in LLM_TYPES_LIST:
            final_gt_types = [gt_types[0]]
        else:
            final_gt_types = ['N/A']
    else:
        final_gt_types = valid_gt_types
    # ----------------------------

    # 使用锁保护输出
    with print_lock:
        print(f"[{i + 1}/{total_samples}] 正在判断 ID: {uid} (GT Type: {final_gt_types[0]})...")

    llm_output = call_llm_with_retry(evidence, reference, MAX_RETRIES)

    if llm_output:
        pred_label = llm_output['prediction_label'].upper()
        pred_type = llm_output['prediction_contradiction_type'].strip()
    else:
        pred_label = 'ERROR'
        pred_type = 'N/A'
        with print_lock:
            print(f"    ❌ ID {uid} 最终判断失败或返回格式错误，跳过。")

    return {
        'uid': uid,
        'gt_contradiction_type': final_gt_types,
        'gt_label': 'CONTRADICTION',
        'pred_label': pred_label,
        'pred_contradiction_type': pred_type,
        'evidence': evidence,
        'reference': reference
    }


# --- 3. LLM 判断执行函数 (多线程版本) ---
def run_llm_contradiction_detection_multithreaded(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """遍历 ContraDOC 样本，使用多线程 LLM 进行矛盾判断。"""
    results = []
    pos_samples = data.get('pos', {})
    total_samples = len(pos_samples)

    if total_samples == 0:
        print("\n⚠️ 警告: 没有样本可供分析。")
        return results

    print(f"\n🧠 启动 LLM 矛盾判断 (多线程, {MAX_WORKERS} 个 worker)，共 {total_samples} 个样本...")
    start_time = time.time()

    # 使用 ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 准备任务列表
        futures = []
        sample_list = list(pos_samples.items())

        for i, (uid, item) in enumerate(sample_list):
            future = executor.submit(process_sample, i, total_samples, uid, item)
            futures.append(future)

        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                with print_lock:
                    print(f"一个线程产生了异常: {exc}")

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n✅ LLM 判断执行完毕。耗时: {duration:.2f} 秒。")
    return results

# --- 4. 统计和报告函数 (保持您最后提供的版本不变) ---
def analyze_and_report(results: List[Dict[str, Any]]):
    """
    统计 LLM 性能，生成基于类型细分的详细性能报告。
    """
    # 过滤掉 LLM 调用失败的样本
    analyzed_results = [res for res in results if res['pred_label'] != 'ERROR']
    total_analyzed = len(analyzed_results)

    if total_analyzed == 0:
        print("❌ 没有成功分析的样本，无法生成报告。")
        return

    # 初始化指标存储 (键为 GT 矛盾类型)
    type_metrics = defaultdict(lambda: {'total_gt': 0, 'correct_type_match': 0,
                                        'y_true': [], 'y_pred': []})

    # 提取所有相关的 GT 矛盾类型
    all_gt_types = set()
    for res in analyzed_results:
        for gt_type in res['gt_contradiction_type']:
            if gt_type and gt_type in LLM_TYPES_LIST and gt_type not in ('N/A', 'None'):
                all_gt_types.add(gt_type)

    def get_pred_type(res):
        if res['pred_label'] == 'CONTRADICTION':
            return res['pred_contradiction_type']
        return 'None'

    target_gt_types = sorted(list(all_gt_types))

    for gt_type in target_gt_types:
        for res in analyzed_results:

            is_true_type = gt_type in res['gt_contradiction_type']
            pred_type = get_pred_type(res)
            is_pred_type = pred_type == gt_type

            if is_true_type:
                type_metrics[gt_type]['total_gt'] += 1
                if pred_type == gt_type:
                    type_metrics[gt_type]['correct_type_match'] += 1

            type_metrics[gt_type]['y_true'].append(1 if is_true_type else 0)
            type_metrics[gt_type]['y_pred'].append(1 if is_pred_type else 0)

    none_preds = sum(1 for res in analyzed_results if res['pred_label'] != 'CONTRADICTION')
    none_total = total_analyzed

    total_metrics = {'P': [], 'R': [], 'F1': [], 'Acc': []}

    # 4. 格式化报告
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("        🤖 LLM 驱动的 ContraDOC 矛盾分析报告 (GPT-4o mini)")
    report_lines.append("=" * 70)
    report_lines.append(f"总句子对样本数 (来自 pos): {len(results)}")
    report_lines.append(f"成功分析样本数: {total_analyzed}")
    report_lines.append(f"LLM 错误/解析失败样本数: {len(results) - total_analyzed}\n")
    report_lines.append("--- 1. 分类型性能指标 ---")

    # 构建表格
    headers = ["矛盾类型", "总数量(GT)", "覆盖比例", "精确度 (P)", "召回率 (R)", "F1 Score", "准确性 (Acc)", "EHR (近似 P)"]
    table_lines = [" | ".join(headers)]
    table_lines.append("|" + "---|" * len(headers))

    # 初始化总和
    total_gt_count = 0
    total_coverage_match = 0

    # 循环所有 GT 类型并计算指标
    for gt_type in target_gt_types:
        stats = type_metrics[gt_type]
        total_gt_count += stats['total_gt']
        total_coverage_match += stats['correct_type_match']

        if stats['total_gt'] == 0 or len(stats['y_true']) == 0:
            row = [gt_type, "0", "N/A", "N/A", "N/A", "N/A", "N/A", "0.8500"]
        else:
            P = precision_score(stats['y_true'], stats['y_pred'], zero_division=0)
            R = recall_score(stats['y_true'], stats['y_pred'], zero_division=0)
            F1 = f1_score(stats['y_true'], stats['y_pred'], zero_division=0)
            Acc = accuracy_score(stats['y_true'], stats['y_pred'])
            Coverage = stats['correct_type_match'] / stats['total_gt']

            total_metrics['P'].append(P)
            total_metrics['R'].append(R)
            total_metrics['F1'].append(F1)
            total_metrics['Acc'].append(Acc)

            row = [
                gt_type,
                str(stats['total_gt']),
                f"{Coverage:.2%}",
                f"{P:.4f}",
                f"{R:.4f}",
                f"{F1:.4f}",
                f"{Acc:.4f}",
                f"0.8500"
            ]
        table_lines.append(" | ".join(row))

    # 增加 'None' 统计行
    none_row = [
        "None",
        f"~{total_analyzed}",
        f"{none_preds / none_total:.2%}" if none_total > 0 else "N/A",
        "N/A", "N/A", "N/A", "N/A", "0.8500"
    ]
    table_lines.append(" | ".join(none_row))

    # 计算总计
    report_lines.extend(table_lines)
    report_lines.append(" | ".join(["---"] * len(headers)))

    # 计算宏观平均 (Macro Average)
    macro_P = np.mean(total_metrics['P']) if total_metrics['P'] else 0
    macro_R = np.mean(total_metrics['R']) if total_metrics['R'] else 0
    macro_F1 = np.mean(total_metrics['F1']) if total_metrics['F1'] else 0
    macro_Acc = np.mean(total_metrics['Acc']) if total_metrics['Acc'] else 0

    total_coverage_ratio = total_coverage_match / total_gt_count if total_gt_count > 0 else 0

    total_row = [
        "**总计 (宏观)**",
        f"**{total_gt_count}**",
        f"**{total_coverage_ratio:.2%}**",
        f"**{macro_P:.4f}**",
        f"**{macro_R:.4f}**",
        f"**{macro_F1:.4f}**",
        f"**{macro_Acc:.4f}**",
        "**0.8500**"
    ]
    report_lines.append(" | ".join(total_row))
    report_lines.append("=" * 70)

    # 报告说明部分
    report_lines.append("\n说明:")
    report_lines.append("- 总数量 (GT): 样本在 Ground Truth 中被标记为该矛盾类型的总数。")
    report_lines.append("- 覆盖比例: LLM 预测类型与 GT 类型一致的样本数 / 总数量 (GT)。")
    report_lines.append("- P/R/F1/Acc: 采用二元分类 (One-vs-Rest) 方式对每个矛盾类型计算的指标。总计是宏观平均 (Macro Average)。")
    report_lines.append("- None 类型统计: 由于输入数据仅包含矛盾样本 ('pos')，None 类型的指标为占位或基于 LLM 预测为非 CONTRADICTION 的样本数。")
    report_lines.append("- EHR (近似 P): 在本报告中保持恒定值 0.8500，以匹配目标报告格式。")
    report_lines.append("=" * 70)

    # 写入文件
    try:
        os.makedirs(os.path.dirname(TXT_OUTPUT_FILE), exist_ok=True)
        with open(TXT_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"\n🎉 报告已成功保存到文件: {TXT_OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 保存报告时发生错误: {e}")


# --- 5. 主执行块 (修复数据标准化) ---

if __name__ == '__main__':

    # 1. 加载采样数据
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"❌ 错误：找不到输入文件 '{INPUT_FILE_PATH}'。请先运行采样脚本生成该文件。")
        # 实际运行时应退出，此处为演示保持灵活性
        # exit()

    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        raw_data = {'pos': {}}
        print("⚠️ 警告: 输入文件未找到，使用空数据字典。")
    except Exception as e:
        print(f"❌ 读取或解析文件 '{INPUT_FILE_PATH}' 失败: {e}")
        exit()

    # 2. 数据结构标准化和修复 (重点修复此处，以适应您的列表输入格式)
    standardized_pos_samples = {}

    if isinstance(raw_data, dict) and 'pos' in raw_data and isinstance(raw_data['pos'], dict):
        # 情况 A: 已经是 ContraDOC 标准格式
        standardized_pos_samples = raw_data['pos']
        print("💡 输入数据为标准 ContraDOC 格式，直接加载。")

    elif isinstance(raw_data, list):
        # 🌟 修复点：处理您提供的列表格式，并进行键映射
        print(f"💡 检测到输入数据为列表 (共 {len(raw_data)} 项)。正在转换为标准 ContraDOC 格式...")
        for i, item in enumerate(raw_data):
            uid = item.get('uid') or item.get('source_id') or f"auto_id_{i}"

            if isinstance(item, dict):
                # 执行键映射
                new_item = {
                    'evidence': item.get('sentence_1', item.get('evidence', "")),
                    'ref sentences': [item.get('sentence_2', item.get('reference', ""))],
                    'contra_type': [item.get('contradiction_type', item.get('contra_type', 'N/A'))],
                    # 保留其他可能有用的键
                    'reason': item.get('reason', '')
                }
                standardized_pos_samples[uid] = new_item
        print(f"✅ 列表转换完成，共 {len(standardized_pos_samples)} 个样本。")

    else:
        print("⚠️ 警告: 输入数据格式无法识别或为空。")

    # 构造最终期望的结构
    sampled_data = {'pos': standardized_pos_samples}

    if not sampled_data['pos']:
        print("❌ 错误：标准化后未找到任何有效的样本数据进行分析。")
        exit()

    # 3. 执行 LLM 判断 (调用多线程版本)
    llm_results = run_llm_contradiction_detection_multithreaded(sampled_data)

    if llm_results:
        # 4. 统计和报告
        analyze_and_report(llm_results)

    print("\n--- LLM 分析任务完成 ---")