import json
import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, Any, List

# --- 配置 ---
INPUT_FILE = "RTE3.xml"
OUTPUT_FILE = "sampled_data.json"
TOTAL_COMPLEX_SAMPLES = 20  # 最终生成的复杂记录总数
MERGE_SIZE = 10  # 每 10 条原始记录融合成一条


# --- 数据加载和解析函数 ---

def load_all_pairs_xml(file_path: str) -> List[Dict[str, str]]:
    """
    解析 XML，将所有 pair 放入一个统一的列表中，不按类型预分类。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return []

    print(f"--- 🚀 正在解析文件: {file_path} ---")
    all_instances = []

    try:
        for event, elem in ET.iterparse(file_path, events=("end",)):
            if event == "end" and elem.tag == "pair":
                entailment = elem.attrib.get("entailment", "UNKNOWN")

                t_text = elem.find('t').text.strip() if elem.find('t') is not None and elem.find('t').text is not None else ""
                h_text = elem.find('h').text.strip() if elem.find('h') is not None and elem.find('h').text is not None else ""

                instance = {
                    "id": elem.attrib.get("id", "N/A"),
                    "entailment": entailment,
                    "task": elem.attrib.get("task", "N/A"),
                    "text": f"{t_text} [SEP] {h_text}"
                }
                all_instances.append(instance)
                elem.clear()

        print(f"✅ 成功加载 {len(all_instances)} 条原始记录。")
        return all_instances

    except Exception as e:
        print(f"🚨 解析失败: {e}")
        return []


# --- 采样与融合函数 ---


def sample_and_merge_fixed_total(all_instances: List[Dict[str, str]],
                                 target_total: int,
                                 merge_size: int) -> List[Dict[str, Any]]:
    """
    从全量数据中随机抽取所需总数，并进行融合。
    """
    # 计算总共需要抽取的原始条数
    required_original_count = target_total * merge_size

    if len(all_instances) < required_original_count:
        actual_total = len(all_instances) // merge_size
        print(f"⚠️ 警告: 原始数据不足以生成 {target_total} 条。将生成最大可能的 {actual_total} 条。")
        sampled_data = all_instances[:actual_total * merge_size]
    else:
        # 随机采样
        sampled_data = random.sample(all_instances, required_original_count)
        print(f"🎯 已从全量数据中随机采样 {required_original_count} 条原始记录。")

    final_complex_samples = []

    # 进行融合
    for i in range(0, len(sampled_data), merge_size):
        batch = sampled_data[i: i + merge_size]

        # 记录该批次中包含的类型分布（方便后续分析）
        types_in_batch = [item["entailment"] for item in batch]

        merged_record = {
            "complex_id": f"COMPLEX_{i // merge_size + 1:03d}",
            "merged_count": merge_size,
            "type_distribution": {t: types_in_batch.count(t) for t in set(types_in_batch)},
            "merged_ids": [item["id"] for item in batch],
            "complex_text": "\n".join([item["text"] for item in batch])
        }
        final_complex_samples.append(merged_record)

    return final_complex_samples


def main():
    # 1. 解析所有数据
    all_pairs = load_all_pairs_xml(INPUT_FILE)
    if not all_pairs:
        return

    # 2. 执行总数 500 的采样融合
    final_list = sample_and_merge_fixed_total(
        all_pairs,
        target_total=TOTAL_COMPLEX_SAMPLES,
        merge_size=MERGE_SIZE
    )

    # 3. 保存
    print(f"\n--- 📊 结果统计 ---")
    print(f"   - 最终生成的复杂记录数: {len(final_list)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 采样完成，结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    random.seed(42)  # 固定随机种子保证结果可复现
    main()