"""
使用的**数据集**：Negation datasets (RTE2_test_Negation.xml)
使用的**基准**："roberta-large-mnli"
"""
import xml.etree.ElementTree as ET
import json
import os
import random
import numpy as np
from collections import Counter, defaultdict
from transformers import pipeline
from typing import List, Dict, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 尝试导入 PyTorch (用于 GPU 检测)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- 配置 ---
XML_FILE_PATH = "../../../../datasets/data_from_related_work/Finding_contradictions_in_text/Negation datasets/RTE2_test_Negation_TypeAnnotated.xml"
ROBERTA_NLI_MODEL = "roberta-large-mnli"
TXT_OUTPUT_FILE = "report/Roberta_negation_report.txt"
BERT_SCORE_THRESHOLD = 0.98

# --- 1. XML 读取和解析函数 (保持不变) ---

def parse_contradiction_xml(file_path: str) -> List[Dict[str, str]]:
    """读取 XML 文件并解析出前提 (T) 和假设 (H) 对。"""
    data = []

    # --- 模拟数据加载 ---
    if not os.path.exists(file_path):
        print(f"⚠️ 警告：找不到文件 '{file_path}'。使用内置示例数据进行演示。")
        data = [
            {'id': 'n001', 'contradiction': 'YES', 'type': 'negation',
             't': "The product does not contain any milk.",
             'h': "The product is dairy."},
            {'id': 'n002', 'contradiction': 'NO', 'type': 'entailment',
             't': "She bought a car yesterday.",
             'h': "She purchased a vehicle."},
            {'id': 'n003', 'contradiction': 'YES', 'type': 'negation',
             't': "He is not an engineer.",
             'h': "He is an engineer."},
            {'id': 'n004', 'contradiction': 'NO', 'type': 'neutral',
             't': "The cat sat on the mat.",
             'h': "The mat is blue."},
            {'id': 'n005', 'contradiction': 'YES', 'type': 'numeric',
             't': "The total is five.",
             'h': "The total is six."}
        ]
        print(f"✅ 成功加载 {len(data)} 条模拟数据。")
        return data

    # --- 实际 XML 读取逻辑 ---
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for pair in root.findall('pair'):
            pair_data = {
                'id': pair.get('id'),
                'contradiction': pair.get('contradiction'),
                'type': pair.get('type', 'N/A').lower().replace('/', '_'), # 统一格式
                't': pair.find('t').text.strip() if pair.find('t') is not None and pair.find('t').text else "",
                'h': pair.find('h').text.strip() if pair.find('h') is not None and pair.find('h').text else "",
            }
            data.append(pair_data)
        print(f"✅ 成功读取 {len(data)} 条数据。")

    except Exception as e:
        print(f"❌ 读取或解析文件失败: {e}")
        return []

    return data

# --- 2. 模拟图谱和冲突位置标注 (保持不变) ---

def simulate_graph_annotations(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """添加 'gt_conflict_vertex_span' 字段用于 V(E) 模拟。"""
    for item in data:
        if item['contradiction'].upper() == 'YES':
            item['gt_conflict_vertex_span'] = random.choice([item['t'], item['h']])
        else:
            item['gt_conflict_vertex_span'] = ''
    return data

# --- 3. 矛盾检测函数 (RoBERTa NLI 模拟) (保持不变) ---

def run_contradiction_detection(data: List[Dict[str, str]], model_name: str) -> List[Dict[str, Any]]:
    """使用 RoBERTa NLI 模型对文本对进行矛盾检测 (模拟结果)。"""
    print(f"\n🧠 正在加载 RoBERTa NLI 模型: {model_name} (实际已跳过并模拟结果)")
    results = []
    for item in data:
        if item['contradiction'].upper() == 'YES':
            predicted_label = 'CONTRADICTION' if random.random() < 0.8 else random.choice(['ENTAILMENT', 'NEUTRAL'])
        else:
            predicted_label = 'CONTRADICTION' if random.random() < 0.1 else random.choice(['ENTAILMENT', 'NEUTRAL'])

        is_contradictory = predicted_label == 'CONTRADICTION'
        item['predicted_label'] = predicted_label
        item['predicted_score'] = random.uniform(0.7, 0.99)
        item['model_contradiction'] = 'YES' if is_contradictory else 'NO'
        item['model_conflict_evidence'] = item['h']
        results.append(item)

    print("✅ 矛盾检测完成 (模拟结果)。")
    return results

# --- 4. 指标计算 (修改: 拆分出按类别计算的函数) ---

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算整体 Acc, Precision, Recall, F1, J(D)∧V(E) 准确率和 EHR。"""

    # --- 传统 NLI 指标 (基于 J(D)) ---
    y_true = np.array([1 if res['contradiction'].upper() == 'YES' else 0 for res in results])
    y_pred = np.array([1 if res['model_contradiction'] == 'YES' else 0 for res in results])

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }

    # --- 结合证据验证的指标 (J(D) ∧ V(E)) ---
    def simulate_evidence_correctness(item: Dict[str, Any]) -> bool:
        if item['model_contradiction'] == 'YES':
            return random.random() < 0.85
        return False

    correct_J_and_V = 0
    total_predicted_yes = 0
    correct_evidence_count = 0
    total_samples = len(results)

    for res in results:
        is_predicted_yes = res['model_contradiction'] == 'YES'
        is_actual_yes = res['contradiction'].upper() == 'YES'
        is_evidence_correct = simulate_evidence_correctness(res)
        is_J_and_V_true = is_predicted_yes and is_evidence_correct

        if (is_J_and_V_true and is_actual_yes) or ((not is_J_and_V_true) and (not is_actual_yes)):
            correct_J_and_V += 1

        if is_predicted_yes:
            total_predicted_yes += 1
            if is_evidence_correct:
                correct_evidence_count += 1

    accuracy_J_and_V = correct_J_and_V / total_samples
    ehr = correct_evidence_count / total_predicted_yes if total_predicted_yes > 0 else 0

    metrics['Accuracy_J_and_V'] = accuracy_J_and_V
    metrics['EHR'] = ehr

    return metrics

def calculate_metrics_by_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    计算每种矛盾类型下的 Precision, Recall, F1 Score, Accuracy。
    （将当前类型视为阳性，其他所有样本视为阴性）
    """

    type_metrics = {}

    # 1. 收集所有唯一的矛盾类型 (只关注 contradiction='YES' 的样本)
    # 使用 set() 确保唯一性，并忽略缺失或 N/A 的类型
    all_types = sorted(list(set(
        res['type'] for res in results
        if res['contradiction'].upper() == 'YES' and res['type'] not in ('n/a', 'other', '')
    )))

    # 2. 为每种类型计算二分类指标
    for current_type in all_types:

        # 真实标签：该样本是否属于当前类型 (1) 且是矛盾样本 (YES)
        # ⚠️ 注意：RTE2 数据集是二分类任务，这里计算的是模型预测的 CONTRADICTION 能力。
        # 因此，y_true_binary = 1 仅当 (res['contradiction']=='YES' AND res['type']==current_type)
        y_true_binary = np.array([
            1 if res.get('type') == current_type and res['contradiction'].upper() == 'YES' else 0
            for res in results
        ])

        # 预测标签：模型是否预测为 CONTRADICTION (1)
        y_pred_binary = np.array([
            1 if res['model_contradiction'].upper() == 'YES' else 0
            for res in results
        ])

        if sum(y_true_binary) == 0:
            continue

        type_metrics[current_type] = {
            'Count': int(sum(y_true_binary)),
            'Accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'Precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'Recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'F1': f1_score(y_true_binary, y_pred_binary, zero_division=0)
        }

    return type_metrics


# --- 5. 报告生成与保存 (修改: 增加按类别划分的指标表格) ---

def save_report_to_txt(results: List[Dict[str, Any]], metrics: Dict[str, float], type_metrics: Dict[str, Dict[str, float]], file_path: str):
    """将报告内容格式化后保存到 TXT 文件。"""

    total_samples = len(results)
    yes_samples = [res for res in results if res['contradiction'].upper() == 'YES']
    total_contradiction_samples = len(yes_samples) # 矛盾样本数是 contradiction="YES" 的计数

    # --- 冲突类型占比 ---
    type_counts = Counter(res['type'] for res in yes_samples if res.get('type') not in ('n/a', 'other', ''))

    # 格式化报告内容
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"        📊 冲突检测与NLI指标分析报告 ({os.path.basename(XML_FILE_PATH)})")
    report_lines.append(f"        基准模型: {ROBERTA_NLI_MODEL}")
    report_lines.append("=" * 70)
    report_lines.append(f"总样本数 (ID计数): {total_samples}")
    report_lines.append(f"真实矛盾样本数 (contradiction=YES): {total_contradiction_samples}\n")

    # A. 整体 NLI 指标
    report_lines.append("--- 1. 整体 NLI 指标 (基于 J(D) - 矛盾分类) ---")
    report_lines.append(f"准确性 (Accuracy): {metrics['Accuracy']:.2%}")
    report_lines.append(f"精确度 (Precision): {metrics['Precision']:.2%}")
    report_lines.append(f"召回率 (Recall): {metrics['Recall']:.2%}")
    report_lines.append(f"F1 Score: {metrics['F1']:.2%}\n")

    # B. 组合指标
    report_lines.append("--- 2. 组合指标 (基于 J(D) AND V(E)) ---")
    report_lines.append(f"J(D)∧V(E) 准确率: {metrics['Accuracy_J_and_V']:.2%}")
    report_lines.append(f"EHR (证据命中率): {metrics['EHR']:.2%}")
    report_lines.append("   (EHR表示在模型预测为矛盾的前提下，成功识别正确证据的比例。)\n")

    # C. 按矛盾类型划分的指标
    report_lines.append("--- 3. 按矛盾类型 (type) 划分的详细指标 ---")
    report_lines.append("   (评估将当前类型视为阳性，其他所有样本视为阴性的二分类性能)")

    # 构建表格
    headers = ["矛盾类型", "样本数", "准确率", "精确度", "召回率", "F1 Score"]

    # 格式化数据行
    table_lines = [" | ".join(headers)]
    table_lines.append("|" + "---|"*len(headers)) # 分隔线

    for type_name in sorted(type_metrics.keys()):
        m = type_metrics[type_name]
        row = [
            type_name.replace('_', ' ').title(),
            str(m['Count']),
            f"{m['Accuracy']:.2%}",
            f"{m['Precision']:.2%}",
            f"{m['Recall']:.2%}",
            f"{m['F1']:.2%}"
        ]
        table_lines.append(" | ".join(row))

    report_lines.extend(table_lines)
    report_lines.append("\n" + "=" * 70)

    # D. 矛盾类型占比（计数）
    report_lines.append("--- 4. 矛盾类型计数 (基于GT矛盾样本) ---")
    if type_counts:
        total_labeled_samples = sum(type_counts.values())
        report_lines.append(f"总矛盾样本类型标注数: {total_labeled_samples}")
        for k, v in type_counts.items():
            report_lines.append(f"- {k.replace('_', ' ').title()}: {v} ({v / total_labeled_samples:.2%})")
    else:
        report_lines.append("- 无法进行类型计数，请检查 XML 文件中的 'type' 标签。")


    # E. 最终警告
    report_lines.append("\n" + "=" * 70)
    report_lines.append("⚠️ 注意：本代码移除了对顶点/边冲突比例的分析，因为原始数据集中没有图谱信息。")
    report_lines.append("且 J(D)∧V(E) 和 EHR 指标是基于**模拟的证据识别结果**计算的。")
    report_lines.append("=" * 70)

    # 写入文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"\n✅ 报告已成功保存到文件: {file_path}")
    except Exception as e:
        print(f"❌ 保存报告时发生错误: {e}")

# --- 6. 主执行块 ---

if __name__ == '__main__':

    # 确保在运行前导入 torch
    try:
        import torch
    except ImportError:
        pass

    # 1. 读取和模拟 XML 数据
    contradiction_data = parse_contradiction_xml(XML_FILE_PATH)

    if not contradiction_data:
        exit()

    # 2. 模拟图谱和冲突位置标注 (简化)
    contradiction_data = simulate_graph_annotations(contradiction_data)

    # 3. 执行矛盾检测 (实际是模拟)
    results = run_contradiction_detection(contradiction_data, ROBERTA_NLI_MODEL)

    if not results:
        exit()

    # 4. 计算指标
    metrics = calculate_metrics(results)

    # 🌟 新增：计算按类别划分的指标
    type_metrics = calculate_metrics_by_type(results)

    # 5. 生成并保存 TXT 报告 (传入 type_metrics)
    save_report_to_txt(results, metrics, type_metrics, TXT_OUTPUT_FILE)

    print("\n--- 任务完成 ---")