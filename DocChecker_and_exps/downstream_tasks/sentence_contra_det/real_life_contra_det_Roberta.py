"""
使用的数据集：Real-life contradiction corpus
使用的基准："roberta-large-mnli"
"""
import xml.etree.ElementTree as ET
from transformers import pipeline
from typing import List, Dict, Tuple, Any
import json

# 假设您的 XML 文件名为 real-life_contradictions.xml
XML_FILE_PATH = "../../../datasets/data_from_related_work/Finding_contradictions_in_text/Real-life contradiction corpus/real-life_contradictions.xml"


# --- 1. XML 读取和解析函数 ---

def parse_contradiction_xml(file_path: str) -> List[Dict[str, str]]:
    """
    读取 XML 文件并解析出前提 (T) 和假设 (H) 对。

    Args:
        file_path: XML 文件路径。

    Returns:
        包含 {id, t, h, contradiction, type} 字典的列表。
    """
    data = []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for pair in root.findall('pair'):
            pair_data = {
                'id': pair.get('id'),
                'contradiction': pair.get('contradiction'),
                'type': pair.get('type'),
                't': pair.find('t').text.strip() if pair.find('t') is not None else "",
                'h': pair.find('h').text.strip() if pair.find('h') is not None else "",
            }
            data.append(pair_data)

        print(f"✅ 成功读取 {len(data)} 条数据。")

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{file_path}'。请确保文件路径正确。")
    except ET.ParseError:
        print(f"❌ 错误：XML 文件 '{file_path}' 解析失败。请检查 XML 格式。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

    return data


# --- 2. 矛盾检测函数 (使用 RoBERTa) ---

def run_contradiction_detection(data: List[Dict[str, str]], model_name: str) -> List[Dict[str, Any]]:
    """
    使用 RoBERTa NLI 模型对文本对进行矛盾检测。

    Args:
        data: 包含文本对的列表 (key: 't', 'h')。
        model_name: Hugging Face NLI 模型的名称。

    Returns:
        包含检测结果的更新列表。
    """
    print(f"\n🧠 正在加载 RoBERTa NLI 模型: {model_name}")

    # 推荐使用 RoBERTa 在 MNLI 上微调的模型，用于 NLI 三分类任务
    # 使用 "text-classification" pipeline，它可以处理多标签分类
    # device=0 表示使用 GPU (如果有的话)，否则使用 CPU (-1)
    try:
        nli_classifier = pipeline(
            "text-classification",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        print(f"❌ 模型加载失败，请检查模型名称或网络连接: {e}")
        return []

    # 准备输入数据格式：NLI 模型通常接受 premise (前提) 和 hypothesis (假设) 的组合
    input_pairs = [
        f"{item['t']} {nli_classifier.tokenizer.sep_token} {item['h']}"
        for item in data
    ]

    print(f"⏳ 正在对 {len(input_pairs)} 对文本进行矛盾检测...")

    # 批量推理，启用填充 (padding) 和截断 (truncation) 以确保批处理成功
    predictions = nli_classifier(
        input_pairs,
        padding=True,
        truncation=True,
        max_length=512,
        batch_size=16  # 可以根据 GPU 显存调整批次大小
    )

    results = []

    for item, pred in zip(data, predictions):
        # NLI 模型的标签通常是 'CONTRADICTION', 'ENTAILMENT', 'NEUTRAL'

        # ⚠️ 关键判断：检测是否为矛盾 (CONTRADICTION)
        is_contradictory = pred['label'].upper() == 'CONTRADICTION'

        item['predicted_label'] = pred['label']
        item['predicted_score'] = float(pred['score'])
        item['model_contradiction'] = 'YES' if is_contradictory else 'NO'
        results.append(item)

    print("✅ 矛盾检测完成。")
    return results

# 定义输出文件名
JSON_OUTPUT_FILE = "roberta_contradiction_results.json"

def save_results_to_json(data: List[Dict[str, Any]], file_path: str):
    """将结果列表保存为 JSON 文件。"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 以便正确保存非 ASCII 字符
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 结果已成功保存到文件: {file_path}")
    except Exception as e:
        print(f"❌ 保存结果时发生错误: {e}")

# --- 3. 主执行块 ---

if __name__ == '__main__':
    # 确保安装了 pytorch 和 transformers 库
    import torch

    # 推荐的 RoBERTa NLI 模型
    ROBERTA_NLI_MODEL = "roberta-large-mnli"
    JSON_OUTPUT_FILE = "roberta_contradiction_results.json"  # 在这里定义文件名

    # 1. 读取 XML 数据
    contradiction_data = parse_contradiction_xml(XML_FILE_PATH)

    if not contradiction_data:
        exit()

    # 2. 执行矛盾检测
    results = run_contradiction_detection(contradiction_data, ROBERTA_NLI_MODEL)

    if not results:
        exit()

    # 3. 展示结果和准确性 (仅对前几个样本)
    print("\n" + "=" * 50)
    print("📋 矛盾检测结果摘要")
    print("=" * 50)

    # 统计准确率 (仅针对文件中的 YES/NO 标签)
    correct_predictions = 0

    for i, res in enumerate(results):

        # 打印前 5 个样本的详细结果
        if i < 5:
            print(f"\n--- 样本 {res['id']} ---")
            print(f"T: {res['t']}")
            print(f"H: {res['h']}")
            print(f"原始标签: {res['contradiction']} (Type: {res['type']})")
            print(f"模型预测: {res['predicted_label']} (Score: {res['predicted_score']:.4f})")
            print(f"模型判断: {'✅ 矛盾' if res['model_contradiction'] == 'YES' else '❌ 非矛盾'} (目标: {res['contradiction']})")

        # 统计准确率（将 NLI 标签 Entailment/Neutral 视为 NO）
        actual_is_contradiction = res['contradiction'].upper() == 'YES'
        model_is_contradiction = res['model_contradiction'].upper() == 'YES'

        if actual_is_contradiction == model_is_contradiction:
            correct_predictions += 1

    # 计算并打印整体准确率
    accuracy = correct_predictions / len(results) if results else 0
    print("\n" + "-" * 50)
    print(f"总样本数: {len(results)}")
    print(f"分类准确率 (Contradiction vs. Non-Contradiction): {accuracy:.2%} ({correct_predictions}/{len(results)})")
    print("-" * 50)

    # 4. 保存结果 (新增步骤)
    save_results_to_json(results, JSON_OUTPUT_FILE)

    # 打印最终总结
    print("\n--- 任务完成 ---")