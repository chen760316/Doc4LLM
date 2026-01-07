import json
import os
import random
from typing import Dict, Any, List, Optional
import xml.etree.ElementTree as ET  # 引入 XML 解析库

# --- 配置 ---
INPUT_FILE = "RTE2_test_Negation.xml"
OUTPUT_FILE = "sampled_data.json"  # 更改输出文件名
TARGET_SAMPLES = 500  # 目标采样数量（不再是复杂记录数）

# 定义常量标签
TARGET_ATTRIBUTE = "contradiction"
TARGET_VALUE = "YES"

# 定义用于连接 T 和 H 的分隔符
SEPARATOR = " [SEP] "


# --- 数据加载和解析函数 ---

def load_and_parse_xml(file_path: str) -> List[Dict[str, str]]:
    """
    加载 XML 文件，解析 <pair> 标签，只提取 contradiction="YES" 的样本。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return []

    print(f"--- 🚀 正在读取和解析文件: {file_path} ---")

    filtered_data: List[Dict[str, str]] = []

    try:
        # 使用 iterparse 逐个解析元素
        for event, elem in ET.iterparse(file_path, events=("end",)):
            if event == "end" and elem.tag == "pair":

                # 检查目标属性是否匹配目标值
                contradiction_val = elem.attrib.get(TARGET_ATTRIBUTE)

                if contradiction_val == TARGET_VALUE:
                    # 提取 <t> 和 <h2> 的文本内容
                    t_text = elem.find('t').text.strip() if elem.find('t') is not None and elem.find('t').text is not None else ""
                    h_text = elem.find('h').text.strip() if elem.find('h') is not None and elem.find('h').text is not None else ""

                    # 构造原始实例
                    instance = {
                        "id": elem.attrib.get("id", "N/A"),
                        "label_text": TARGET_VALUE,
                        # 核心：合并 <t> 和 <h2>
                        "text": f"{t_text}{SEPARATOR}{h_text}"
                    }
                    filtered_data.append(instance)

                # 清理元素，释放内存
                elem.clear()

        print(f"✅ 成功解析数据。")
        print(f"   - 找到 '{TARGET_ATTRIBUTE}={TARGET_VALUE}' 总实例数: {len(filtered_data)}")

        return filtered_data

    except ET.ParseError as e:
        print(f"🚨 错误：XML 解析失败。错误信息: {e}")
        return []
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return []


# --- 采样函数 ---

def sample_data(all_data: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
    """
    从所有符合条件的记录中随机采样指定数量的样本。
    """
    num_available = len(all_data)

    if num_available == 0:
        print(f"⚠️ 警告：没有可用数据进行采样。")
        return []

    num_to_sample = min(target_size, num_available)

    # 随机采样
    sampled_list = random.sample(all_data, num_to_sample)

    print(f"\n--- 🔄 正在进行采样 ---")
    print(f"   - 总可用样本数: {num_available}")
    print(f"   - 实际采样样本数: {len(sampled_list)}")

    return sampled_list


def main():
    # 1. 加载和解析数据，只提取 contradiction="YES"
    # 注意：我们将不再使用融合函数，而是直接进行原始数据采样
    all_contradiction_data = load_and_parse_xml(INPUT_FILE)
    if not all_contradiction_data:
        return

    # 2. 采样 100 条记录
    sampled_list = sample_data(all_contradiction_data, TARGET_SAMPLES)

    # 3. 格式化输出并保存
    num_sampled = len(sampled_list)

    print("\n--- 🎉 采样结果统计 ---")
    print(f"   - 最终总采样记录数: {num_sampled}")

    # 转换为 JSON 格式字符串
    final_json_output = json.dumps(sampled_list, indent=2, ensure_ascii=False)

    # 4. 保存结果
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(final_json_output)
        print(f"\n✅ 采样数据已保存到: {OUTPUT_FILE}")
    except Exception as e:
        print(f"🚨 错误：保存文件 {OUTPUT_FILE} 失败: {e}")

    # 打印预览
    print("\n[结构化采样数据预览 (前 2 个记录)]:")
    if sampled_list:
        print(json.dumps(sampled_list[:2], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 设置随机种子以保证采样的可复现性
    random.seed(42)
    main()