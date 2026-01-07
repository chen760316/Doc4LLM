import json
import os
import random
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET  # 引入 XML 解析库

# --- 配置 ---
INPUT_FILE = "RTE3.xml"
OUTPUT_FILE = "sampled_data.json"
TARGET_SAMPLES_PER_TYPE = 500  # 每种类型最终采样的复杂记录数
MERGE_SIZE = 10  # 每 10 条原始记录融合成一条复杂记录


# --- 数据加载和解析函数 ---

def load_and_parse_xml(file_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    加载 RTE3.xml 文件，解析 <pair> 标签，并按 entailment 属性分类。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return {}

    print(f"--- 🚀 正在读取和解析文件: {file_path} ---")

    data_by_entailment: Dict[str, List[Dict[str, str]]] = {
        "ENTAILMENT": [],
        "CONTRADICTION": [],
        "UNKNOWN": []
    }

    try:
        # 使用 iterparse 逐个解析元素，处理大文件更高效
        for event, elem in ET.iterparse(file_path, events=("end",)):
            if event == "end" and elem.tag == "pair":
                entailment = elem.attrib.get("entailment")

                if entailment in data_by_entailment:
                    # 提取 <t> 和 <h2> 的文本内容
                    t_text = elem.find('t').text.strip() if elem.find('t') is not None and elem.find('t').text is not None else ""
                    h_text = elem.find('h').text.strip() if elem.find('h') is not None and elem.find('h').text is not None else ""

                    # 构造原始实例
                    instance = {
                        "id": elem.attrib.get("id", "N/A"),
                        "entailment": entailment,
                        "task": elem.attrib.get("task", "N/A"),
                        # 💥 新增属性：合并 <t> 和 <h2>
                        "text": f"{t_text} [SEP] {h_text}"
                    }
                    data_by_entailment[entailment].append(instance)

                # 清理元素，释放内存
                elem.clear()

        print(f"✅ 成功解析数据。")
        for k, v in data_by_entailment.items():
            print(f"   - {k} 总实例数: {len(v)}")

        return data_by_entailment

    except ET.ParseError as e:
        print(f"🚨 错误：XML 解析失败。错误信息: {e}")
        return {}
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return {}


# --- 采样和融合函数 ---

def merge_and_sample_complex(data_by_entailment: Dict[str, List[Dict[str, str]]],
                             target_complex_size: int,
                             merge_size: int) -> List[Dict[str, Any]]:
    """
    按类型采样原始数据，然后每 MERGE_SIZE 条融合成一条复杂记录。
    """
    final_complex_samples: List[Dict[str, Any]] = []

    # 💥 计算每种类型需要的原始采样数
    target_original_size = target_complex_size * merge_size

    print(f"\n--- 🔄 正在进行采样和融合 (每类型目标复杂记录: {target_complex_size} 条) ---")

    for entailment_type, instances in data_by_entailment.items():

        # 1. 采样原始数据
        num_available = len(instances)
        # 确保有足够的实例进行采样
        if num_available < target_original_size:
            print(f"⚠️ 警告: {entailment_type} 实例数不足 {target_original_size}。使用所有 {num_available} 条。")
            sampled_instances = instances
        else:
            sampled_instances = random.sample(instances, target_original_size)

        # 2. 融合创建复杂记录
        merged_list: List[Dict[str, Any]] = []

        # 遍历采样后的列表，每 merge_size 条进行切片
        for i in range(0, len(sampled_instances), merge_size):
            batch = sampled_instances[i:i + merge_size]

            # 如果批次大小不足 merge_size，则跳过最后不完整的批次
            if len(batch) != merge_size:
                continue

            # 融合逻辑
            merged_record = {
                "complex_id": f"{entailment_type}_{i // merge_size}",
                "complex_entailment": entailment_type,
                "merged_count": merge_size,
                "merged_ids": [item["id"] for item in batch],
                "merged_tasks": [item["task"] for item in batch],
                # 💥 核心融合：将所有 text 属性用换行符连接起来
                "complex_text": "\n".join([item["text"] for item in batch])
            }
            merged_list.append(merged_record)

        final_complex_samples.extend(merged_list)
        print(f"   - {entailment_type} 最终复杂记录数: {len(merged_list)}")

    return final_complex_samples


def main():
    # 1. 加载和解析数据
    data_by_entailment = load_and_parse_xml(INPUT_FILE)
    if not data_by_entailment:
        return

    # 2. 采样和融合
    sampled_complex_list = merge_and_sample_complex(
        data_by_entailment,
        target_complex_size=TARGET_SAMPLES_PER_TYPE,
        merge_size=MERGE_SIZE
    )

    # 3. 格式化输出并保存
    num_sampled = len(sampled_complex_list)

    print("\n--- 🎉 采样和融合结果统计 ---")
    print(f"   - 最终总复杂记录数: {num_sampled}")

    # 转换为 JSON 格式字符串
    final_json_output = json.dumps(sampled_complex_list, indent=2, ensure_ascii=False)

    # 4. 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_json_output)

    print(f"\n✅ 复杂采样数据已保存到: {OUTPUT_FILE}")
    print("\n[结构化采样数据预览 (前 2 个复杂记录)]:")

    # 打印预览
    preview_lines = final_json_output.split('\n')
    print('\n'.join(preview_lines[:min(len(preview_lines), 40)]) + "\n...")


if __name__ == "__main__":
    # 设置随机种子以保证采样的可复现性（可选）
    random.seed(42)
    main()