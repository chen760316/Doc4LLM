import json
import os
import random
from typing import List, Dict, Any, Optional

# --- 配置 ---
INPUT_FILE = "data.json"  # 假设您的完整数据集文件名为 data.json
OUTPUT_FILE = "sampled_data.json"
SAMPLE_SIZE = 100  # 目标采样数量
random.seed(42)  # 设置随机种子以确保结果可复现


# --- 数据加载函数 ---

def load_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    尝试从 JSON 文件中加载数据。
    假设文件是 [List of instances] 或 {Domain: [List of instances]}。
    如果发现是字典结构，则平铺为列表。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return None

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # 案例 1: 文件是一个列表（最常见的格式）
            print(f"✅ 文件结构为列表。总实例数: {len(data)}")
            return data

        elif isinstance(data, dict):
            # 案例 2: 文件是领域字典 (如您第一个提问中所示的领域结构)
            all_instances = []
            for domain, instances_list in data.items():
                if isinstance(instances_list, list):
                    all_instances.extend(instances_list)
            print(f"✅ 文件结构为字典，已平铺。总实例数: {len(all_instances)}")
            return all_instances

        else:
            print("🚨 错误：文件内容结构既非列表也非字典。")
            return None

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None


# --- 采样函数 ---

def sample_instances(all_instances: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
    """
    从所有实例中随机采样指定数量的实例。
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
    # 1. 加载数据
    all_instances = load_data(INPUT_FILE)
    if all_instances is None:
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

    # 打印前 40 行预览
    preview_lines = final_json_output.split('\n')
    print('\n'.join(preview_lines[:min(len(preview_lines), 40)]) + "\n...")


if __name__ == "__main__":
    main()