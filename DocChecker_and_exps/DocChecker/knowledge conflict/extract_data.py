import json
import os
import random
from typing import Dict, Any, List, Optional

# --- 配置 ---
INPUT_FILE = "data.json"
SAMPLE_SIZE = 100  # 目标采样数量


# --- 数据加载函数 ---

def load_data(file_path: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """尝试从 JSON 文件中加载数据，预期结构为 {Domain: [List of instances]}。"""
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return None

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print("🚨 错误：文件内容结构不符合预期的顶层字典格式。")
            return None

        # 简单检查子结构是否是列表
        if not all(isinstance(v, list) for v in data.values()):
            print("⚠️ 警告：顶层键的值不是列表，可能与预期结构不符。")

        print(f"✅ 成功加载数据。包含 {len(data)} 个顶级领域。")
        return data

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None


# --- 采样函数 ---

def sample_instances(data: Dict[str, List[Dict[str, Any]]], sample_size: int) -> List[Dict[str, Any]]:
    """
    遍历所有领域，平铺所有实例，组合冲突文本，然后随机采样指定数量的实例。
    """
    all_instances: List[Dict[str, Any]] = []

    # 1. 收集所有实例并处理新属性
    for domain, instances_list in data.items():
        if not isinstance(instances_list, list):
            continue

        for instance in instances_list:

            # 确保关键字段存在
            if all(key in instance for key in ["question_about_conflicting_segments", "parametric_knowledge", "conflicting_knowledge"]):

                # --- 🎯 新增属性组合逻辑 ---
                instance['conflicting_texts'] = [
                    instance["question_about_conflicting_segments"],
                    instance["parametric_knowledge"],
                    instance["conflicting_knowledge"]
                ]
                # -----------------------------

                # 添加领域信息
                instance['source_domain'] = domain
                all_instances.append(instance)
            else:
                print(f"⚠️ 警告: 实例缺少关键字段，跳过。领域: {domain}")

    total_available = len(all_instances)
    num_to_sample = min(sample_size, total_available)

    if total_available == 0:
        print("⚠️ 警告：未找到任何实例，无法采样。")
        return []

    # 2. 随机采样
    sampled_list = random.sample(all_instances, num_to_sample)

    print(f"   - 总可用实例数 (含冲突文本): {total_available}")
    print(f"   - 实际采样数: {len(sampled_list)}")

    # 3. (可选) 清理原始字段以减少冗余，如果不需要原始字段的话。
    # 这里保持不变，保留原始字段，只新增 conflicting_texts。

    return sampled_list


def main():
    # 1. 加载数据
    data_by_domain = load_data(INPUT_FILE)
    if data_by_domain is None:
        return

    # 2. 采样 100 个实例
    sampled_instances_list = sample_instances(data_by_domain, SAMPLE_SIZE)

    # 3. 格式化输出
    num_sampled = len(sampled_instances_list)

    print("\n--- 🎉 采样结果统计 ---")
    print(f"   - 最终总采样数: {num_sampled}")

    # 转换为 JSON 格式字符串
    final_json_output = json.dumps(sampled_instances_list, indent=2, ensure_ascii=False)

    # 4. 打印/保存结果
    output_filename = "sampled_data.json"  # 修改输出文件名以区分
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_json_output)

    print(f"\n✅ 采样数据已保存到: {output_filename}")
    print("\n[结构化采样数据预览 (前 2 个实例，包含 conflicting_texts)]:")

    # 打印前 40 行预览
    preview_lines = final_json_output.split('\n')
    print('\n'.join(preview_lines[:min(len(preview_lines), 40)]) + "\n...")


if __name__ == "__main__":
    main()