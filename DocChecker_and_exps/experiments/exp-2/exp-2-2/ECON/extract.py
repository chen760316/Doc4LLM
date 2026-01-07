import json
import os
import random
from typing import Dict, List, Any, Tuple

# --- 配置文件路径 ---
# 假设 sampled_ECON.json 包含冲突数据 (Positives)
POS_SOURCE_PATH = "sampled_ECON.json"
# 假设 ContraDoc.json 包含非冲突数据 (Negatives)
NEG_SOURCE_PATH = "ContraDoc.json"
OUTPUT_PATH = "sampled_data.json"
# 负例采样目标数量
TARGET_NEG_SAMPLE_SIZE = 50


# --- 1. 数据加载与提取函数 (正例部分 - 已修改) ---

def load_and_extract_positives(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    从 sampled_ECON.json 文件中加载数据，提取 'evidence' 字段作为正例文本。
    字典格式: {unique_id: {text: ..., label: 1, ...}}
    """
    pos_data_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 正例源文件未找到: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {file_path} JSON 解析失败。")
        return {}

    # 假设文件顶层是一个包含多个案例的列表
    if not isinstance(data_list, list):
        print(f"❌ 警告: {file_path} 预期为列表，但加载为 {type(data_list).__name__}。跳过提取。")
        return {}

    for i, item in enumerate(data_list):
        # 修正逻辑：检查 'evidence' 字段是否存在，并且 'facts' 字段也存在（确保是有效案例）
        if "evidence" in item and item["evidence"] and "facts" in item:
            # 使用 'evidence' 字段作为完整的文档文本
            full_text = item["evidence"]
            unique_id = f"pos_{i}"

            # 创建文档详情字典
            doc_details = {
                "text": full_text,
                "label": 1,  # 矛盾标签
                "source_domain": item.get("source_domain", "Unknown"),
                "qid": item.get("qid") # 保留qid方便追踪
            }
            pos_data_dict[unique_id] = doc_details

    print(f"✅ 从 {file_path} 中提取 {len(pos_data_dict)} 个正例。")
    return pos_data_dict


# --- 2. 负例加载与采样函数 (与上次版本一致) ---

def load_and_sample_negatives(file_path: str, sample_size: int) -> Dict[str, Dict[str, Any]]:
    """
    从 ContraDoc.json 文件中加载 'neg' 键下的所有文本，随机采样指定数量，并返回一个字典。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 负例源文件未找到: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ 错误: 文件 {file_path} JSON 解析失败。")
        return {}

    # 预期文件顶层是一个包含 'neg' 键的字典
    neg_data = full_data.get("neg", {})

    if not isinstance(neg_data, dict):
        print(f"❌ 警告: {file_path} 中的 'neg' 预期为字典，但加载为 {type(neg_data).__name__}。跳过提取。")
        return {}

    # 获取所有负例的 unique_id 列表
    all_neg_ids = list(neg_data.keys())
    total_neg_count = len(all_neg_ids)

    # 确定实际采样数量
    actual_sample_size = min(sample_size, total_neg_count)

    # 随机采样 unique_id
    if actual_sample_size < total_neg_count:
        sampled_ids = random.sample(all_neg_ids, actual_sample_size)
    else:
        # 如果总数小于或等于目标数量，则全部使用
        sampled_ids = all_neg_ids

    sampled_neg_dict = {}
    for unique_id in sampled_ids:
        doc_info = neg_data[unique_id]
        if "text" in doc_info:
            # 负例标签为 0
            doc_details = {
                "text": doc_info["text"],
                "label": 0,  # 非矛盾标签
                "doc_type": doc_info.get("doc_type", "Unknown")
            }
            sampled_neg_dict[unique_id] = doc_details

    print(f"✅ 从 {file_path} (总数: {total_neg_count}) 中随机采样 {len(sampled_neg_dict)} 个负例。")
    return sampled_neg_dict


# --- 3. 主执行逻辑 ---

if __name__ == "__main__":

    # 1. 提取正例 (Positives)
    positives_dict = load_and_extract_positives(POS_SOURCE_PATH)

    # 2. 提取并采样负例 (Negatives)
    negatives_dict = load_and_sample_negatives(NEG_SOURCE_PATH, TARGET_NEG_SAMPLE_SIZE)

    # 3. 组织最终的数据结构
    final_structured_data = {
        "pos": positives_dict,
        "neg": negatives_dict
    }

    # 4. 保存结果
    if positives_dict or negatives_dict:
        try:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_structured_data, f, ensure_ascii=False, indent=4)

            total_pos = len(positives_dict)
            total_neg = len(negatives_dict)

            print("\n--- 结果汇总 ---")
            print(f"✨ 成功生成新数据集，总样本数: {total_pos + total_neg}")
            print(f"   正例 (矛盾): {total_pos} (位于 'pos' 键下)")
            print(f"   负例 (非矛盾): {total_neg} (位于 'neg' 键下, 目标采样 {TARGET_NEG_SAMPLE_SIZE} 条)")
            print(f"💾 数据已保存到: {OUTPUT_PATH}")
        except IOError as e:
            print(f"❌ 错误: 写入文件失败: {e}")
    else:
        print("\n❌ 警告: 没有提取到有效数据，未生成输出文件。")