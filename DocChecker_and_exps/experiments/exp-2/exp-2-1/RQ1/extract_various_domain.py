import json
import random
import os
from collections import defaultdict
from typing import Dict, List, Any

# --- 配置 ---
# ⚠️ 请根据您的实际文件路径修改
INPUT_FILE_PATH = "ContraDoc.json"
# 输出文件名
OUTPUT_FILE_PATH = "sampled_by_doc_type.json"
# 目标总采样数量
TARGET_TOTAL_SAMPLES = 99
# 设置随机种子以保证结果可复现
random.seed(42)


def sample_contradoc_by_doc_type(input_path: str, output_path: str, total_count: int):
    """
    从 ContraDOC JSON 文件的 'pos' 部分，按 'doc_type' 均匀采样指定数量的样本。
    """
    if not os.path.exists(input_path):
        print(f"❌ 错误：找不到输入文件 '{input_path}'。请检查路径。")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ 错误：JSON 文件 '{input_path}' 解析失败。")
        return
    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")
        return

    # 1. 初始化存储结构：按 doc_type 分组
    # 键是 doc_type (如 'story', 'wiki', 'news')，值是符合该类型的样本列表
    doc_type_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # 提取 pos 样本
    pos_samples = full_data.get('pos', {})

    # 2. 按 doc_type 分组样本
    print(f"🔍 正在处理 {len(pos_samples)} 条正例样本...")

    for uid, item in pos_samples.items():
        doc_type = item.get('doc_type')

        if doc_type:
            doc_type_tag = doc_type.lower().strip()  # 标准化标签

            # 存储样本 ID 和内容
            sample_data = item.copy()
            sample_data['uid'] = uid
            doc_type_groups[doc_type_tag].append(sample_data)

    all_doc_types = sorted(list(doc_type_groups.keys()))
    num_doc_types = len(all_doc_types)

    if num_doc_types == 0:
        print("⚠️ 警告：未找到任何有效的 'doc_type' 标签。任务终止。")
        return

    # 3. 计算每种 doc_type 应采样的数量 (均匀分配)

    # 基础数量
    base_count_per_type = total_count // num_doc_types
    # 剩余数量 (用于分配给前几个类型)
    remainder = total_count % num_doc_types

    samples_per_type = {}
    for i, doc_type in enumerate(all_doc_types):
        # 将余数分配给靠前的类型
        count = base_count_per_type + (1 if i < remainder else 0)
        samples_per_type[doc_type] = count

    print(f"\n📦 发现以下 {num_doc_types} 种 doc_type: {', '.join(all_doc_types)}")
    print(f"📦 目标总采样 {total_count} 条，分配给每种类型: {samples_per_type}")

    # 4. 从每个 doc_type 组中进行采样
    sampled_uids = set()
    sampled_data_dict: Dict[str, Any] = {}
    current_total_samples = 0

    for doc_type_tag in all_doc_types:
        samples = doc_type_groups[doc_type_tag]
        count = samples_per_type[doc_type_tag]

        if len(samples) < count:
            # 如果样本不足，则全部采样
            print(f"⚠️ 警告：类型 '{doc_type_tag}' 样本不足 ({len(samples)} < {count})，将全部采样。")
            sampled_list = samples
        else:
            # 随机采样
            sampled_list = random.sample(samples, count)

        # 将采样的样本添加到最终结果字典中
        for sample in sampled_list:
            uid = sample.pop('uid')  # 移除临时 uid
            if uid not in sampled_uids:
                sampled_uids.add(uid)
                sampled_data_dict[uid] = sample
                current_total_samples += 1
            # 注意：此处没有处理重复 UID 的情况，因为 pos 样本的 UID 本身应该是唯一的。

        print(f"✅ 类型 '{doc_type_tag}' 采样完成 ({len(sampled_list)} 条)。")

    # 5. 封装结果并保存为 JSON 文件

    final_output = {
        "pos": sampled_data_dict,
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"\n🎉 成功将 {current_total_samples} 条样本保存到文件: {output_path}")
        print(f"🎉 最终 doc_type 分布:")
        # 简单统计最终分布
        final_doc_types = defaultdict(int)
        for item in sampled_data_dict.values():
            final_doc_types[item.get('doc_type', 'N/A').lower()] += 1
        print(dict(final_doc_types))

    except Exception as e:
        print(f"❌ 保存文件时发生错误: {e}")


# --- 主执行块 ---
if __name__ == '__main__':
    sample_contradoc_by_doc_type(INPUT_FILE_PATH, OUTPUT_FILE_PATH, TARGET_TOTAL_SAMPLES)