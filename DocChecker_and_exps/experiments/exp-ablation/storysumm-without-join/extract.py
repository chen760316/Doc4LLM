import json
import os
import random
from typing import List, Dict, Any, Optional

# --- 配置 ---
INPUT_FILE = "storysumm.json"  # 更改为目标输入文件
OUTPUT_FILE = "sampled_data.json"
SAMPLE_SIZE = 100  # 目标采样数量
random.seed(42)  # 设置随机种子以确保结果可复现


# --- 数据加载函数 ---

def load_data(file_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    尝试从 JSON 文件中加载数据。
    假设文件是 {Key: Instance_Dictionary} 的字典结构。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return None

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            print(f"✅ 文件结构为字典 (Key: Instance)。总实例数: {len(data)}")
            return data

        else:
            print("🚨 错误：文件内容结构不是预期的 {Key: Instance_Dictionary} 格式。")
            return None

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None


# --- 数据处理和采样函数 ---

def process_and_sample_instances(
        all_instances_dict: Dict[str, Dict[str, Any]],
        sample_size: int
) -> List[Dict[str, Any]]:
    """
    从所有实例中随机采样指定数量的实例，并根据要求处理数据。

    **关键修正：** 1. 遵循 errors=0 提取 summary 句子并合并 explanations。
    2. **如果生成的 'text' 字段为空，则在最终输出中忽略该案例**（解决了案例 10/12 的问题）。
    """
    all_keys = list(all_instances_dict.keys())
    total_available = len(all_keys)

    if total_available == 0:
        print("⚠️ 警告：未找到任何实例，无法采样。")
        return []

    # 尝试采样，直到达到 SAMPLE_SIZE 或遍历完所有可用的键
    sampled_keys = random.sample(all_keys, min(sample_size * 2, total_available))  # 初始多采一点，用于过滤
    processed_list = []

    attempts = 0
    max_attempts = len(all_keys)

    # 使用一个 while 循环来确保我们过滤掉空的 text 结果，并尝试达到 SAMPLE_SIZE
    while len(processed_list) < sample_size and attempts < max_attempts:

        # 如果初始采样的池子用完了，但还没达到目标数量，就尝试随机选择剩余的键
        if not sampled_keys:
            remaining_keys = [k for k in all_keys if k not in [item['original_key'] for item in processed_list]]
            if not remaining_keys:
                break  # 没有更多键了
            key_to_process = random.choice(remaining_keys)
        else:
            key_to_process = sampled_keys.pop(0)

        instance = all_instances_dict[key_to_process]

        summary: List[str] = instance.get("summary", [])
        errors: List[int] = instance.get("errors", [])
        explanations: List[str] = instance.get("explanations", [])
        story_id: Any = instance.get("story-id", key_to_process)

        # 1. 筛选 summary 中 errors 为 0 的句子
        correct_summary_sentences = []
        if len(summary) == len(errors):
            for sentence, error_flag in zip(summary, errors):
                if error_flag == 0:
                    correct_summary_sentences.append(sentence)

        # 2. 组合文本
        combined_text_parts = []

        if correct_summary_sentences:
            combined_text_parts.append("Summary parts with no errors:")
            combined_text_parts.extend([f"- {s}" for s in correct_summary_sentences])

        # 仅在 explanations 非空时添加标题
        if explanations:
            if correct_summary_sentences:
                combined_text_parts.append("")  # 添加空行分隔
            combined_text_parts.append("Explanations (Corrections):")
            combined_text_parts.extend([f"- {e}" for e in explanations])

        final_text = "\n".join(combined_text_parts).strip()

        # 3. 检查并添加结果：只有当生成的 text 非空时才添加
        if final_text:
            processed_list.append({
                "story-id": story_id,
                "text": final_text,
                "original_key": key_to_process  # 用于跟踪，非最终输出
            })

        attempts += 1

    final_output = [{k: v for k, v in item.items() if k != 'original_key'} for item in processed_list[:sample_size]]

    print(f"   - 总可用实例数: {total_available}")
    print(f"   - 实际采样子集数: {len(final_output)}")

    return final_output


def main():
    # 1. 加载数据
    all_instances_dict = load_data(INPUT_FILE)
    if all_instances_dict is None:
        return

    # 2. 采样并处理实例
    processed_sampled_list = process_and_sample_instances(all_instances_dict, SAMPLE_SIZE)

    # 3. 格式化输出
    num_sampled = len(processed_sampled_list)

    print("\n--- 🎉 采样结果统计 ---")
    print(f"   - 最终总采样数: {num_sampled}")

    if num_sampled == 0:
        print("⚠️ 警告：没有实例包含有效的（非空）text字段。")
        return

    # 转换为 JSON 格式字符串
    final_json_output = json.dumps(processed_sampled_list, indent=2, ensure_ascii=False)

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