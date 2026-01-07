import json
import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple

# --- 配置 ---
INPUT_FILE = "real-life_contradictions.xml"  # 保持 XML 文件名
OUTPUT_FILE = "sampled_data.json"
SAMPLE_SIZE = 131  # 目标采样数量
random.seed(42)  # 设置随机种子以保证结果可复现


# --- XML 数据加载与解析函数 ---

def load_and_parse_xml(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 XML 文件，解析 <pair> 标签，并提取属性和文本内容。
    将 <t> 和 <h> 文本合并为 'text' 字段。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return []

    print(f"--- 🚀 正在读取和解析 XML 文件: {file_path} ---")

    parsed_instances: List[Dict[str, Any]] = []
    # 定义合并后的文本分隔符
    TEXT_HYPOTHESIS_SEPARATOR = " [SEP] "

    try:
        # 使用 UTF-8 编码打开文件
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 遍历所有 <pair> 标签
        for pair_element in root.findall(".//pair"):

            # 提取属性
            pair_id = pair_element.get('id')
            contradiction_status = pair_element.get('contradiction')
            contradiction_type = pair_element.get('type')

            # 提取 <t> 和 <h> 文本
            t_element = pair_element.find('t')
            h_element = pair_element.find('h')

            t_text = t_element.text.strip() if t_element is not None and t_element.text else ""
            h_text = h_element.text.strip() if h_element is not None and h_element.text else ""

            if t_text and h_text:
                # 🎯 核心逻辑：将 t 和 h 合并为 text 字段
                merged_text = f"{t_text}{TEXT_HYPOTHESIS_SEPARATOR}{h_text}"

                parsed_instances.append({
                    "id": pair_id,
                    "contradiction_status": contradiction_status,
                    "contradiction_type": contradiction_type,
                    "text": merged_text,
                    # 可以选择保留原始字段，但根据要求只保留合并后的 text
                    # "t": t_text,
                    # "h": h_text
                })

        print(f"✅ XML 解析成功。总共找到 {len(parsed_instances)} 条记录。")
        return parsed_instances

    except ET.ParseError as e:
        print(f"🚨 错误：XML 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return []
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return []


# --- 采样函数 (适配扁平化列表) ---

def sample_instances(all_instances: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    """
    从所有实例列表中随机采样指定数量的实例。
    """
    total_available = len(all_instances)
    num_to_sample = min(sample_size, total_available)

    if total_available == 0:
        print("⚠️ 警告：未找到任何实例，无法采样。")
        return []

    # 随机采样
    sampled_list = random.sample(all_instances, num_to_sample)

    print(f"   - 总可用实例数: {total_available}")
    print(f"   - 实际采样数: {len(sampled_list)}")

    return sampled_list


def main():
    # 1. 加载和解析 XML 数据
    all_instances = load_and_parse_xml(INPUT_FILE)
    if not all_instances:
        return

    # 2. 采样 100 个实例
    sampled_instances_list = sample_instances(all_instances, SAMPLE_SIZE)

    # 3. 格式化输出
    num_sampled = len(sampled_instances_list)

    print("\n--- 🎉 采样结果统计 ---")
    print(f"   - 最终总采样数: {num_sampled}")

    # 转换为 JSON 格式字符串
    final_json_output = json.dumps(sampled_instances_list, indent=2, ensure_ascii=False)

    # 4. 打印/保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_json_output)

    print(f"\n✅ 采样数据已保存到: {OUTPUT_FILE}")
    print("\n[结构化采样数据预览 (前 2 个实例)]:")

    # 打印预览
    preview_lines = final_json_output.split('\n')
    print('\n'.join(preview_lines[:min(len(preview_lines), 20)]) + "\n...")


if __name__ == "__main__":
    # 确保运行环境有 xml.etree.ElementTree 库 (标准库自带)
    main()