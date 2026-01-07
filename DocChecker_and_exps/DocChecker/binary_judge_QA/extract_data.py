import json
import random
import os
from typing import Dict, Any, List, Optional

# --- 配置 ---
INPUT_FILE = "Boolq.jsonl"
OUTPUT_FILE = "sampled_data.json"
SAMPLE_SIZE = 100  # 目标采样数量
random.seed(42)  # 设置随机种子以保证结果可复现

# 定义用于连接不同证据段落的分隔符
SEPARATOR = "\n\n--- OPPOSING EVIDENCE ---\n\n"


# --- 数据处理函数 ---

def load_and_sample_data(file_path: str, sample_size: int) -> List[Dict[str, Any]]:
    """
    加载 JSONL 文件，并从中随机采样指定数量的案例。
    """
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return []

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    all_data = []

    # 1. 逐行读取 JSONL 文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    all_data.append(data)
                except json.JSONDecodeError:
                    print(f"⚠️ 警告：跳过一行格式错误的 JSONL 数据: {line.strip()[:50]}...")
                    continue
    except Exception as e:
        print(f"🚨 读取文件时发生错误: {e}")
        return []

    total_available = len(all_data)
    num_to_sample = min(sample_size, total_available)

    if total_available == 0:
        print("⚠️ 警告：文件中没有数据可供采样。")
        return []

    # 2. 随机采样
    sampled_list = random.sample(all_data, num_to_sample)

    print(f"✅ 成功找到 {total_available} 条记录。")
    print(f"✅ 实际采样 {len(sampled_list)} 条记录。")

    return sampled_list


def construct_contradiction_text(sampled_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    遍历采样的案例，将 'source' 和 'adv' 中的所有论点合并成一个 'text' 字段。
    """
    processed_list = []

    for instance in sampled_data:
        # 1. 获取正确的证据 (Source)
        source_text = instance.get("source", "").strip()

        # 2. 收集所有反对证据 (Adversarial Arguments)
        adv = instance.get("adv", {})

        opposing_arguments = []
        if adv:
            # a. 控制结论 (Control statement)
            control = adv.get("control", "").strip()
            if control:
                opposing_arguments.append(f"Control: {control}")

            # b. 逻辑论证 (Logical)
            for arg in adv.get("logical", []):
                if arg.strip():
                    opposing_arguments.append(f"Logical: {arg.strip()}")

            # c. 信誉论证 (Credibility)
            for arg in adv.get("credibility", []):
                if arg.strip():
                    opposing_arguments.append(f"Credibility: {arg.strip()}")

            # d. 情感论证 (Emotional)
            for arg in adv.get("emotional", []):
                if arg.strip():
                    opposing_arguments.append(f"Emotional: {arg.strip()}")

        # 3. 合并文本

        # 构造反对证据的整合文本
        opposing_text = "\n".join(opposing_arguments).strip()

        if source_text and opposing_text:
            # 基础证据 (Source) + 分隔符 + 所有反对证据 (Adv)
            final_text = f"Primary Evidence: {source_text}{SEPARATOR}{opposing_text}"
        elif source_text:
            final_text = f"Primary Evidence: {source_text}"
        elif opposing_text:
            final_text = opposing_text
        else:
            final_text = ""

        # 4. 构造新的 JSON 对象

        # 保留问题和答案，以及构造的 text 字段
        new_instance = {
            "question": instance.get("question", ""),
            "answer": instance.get("answer", False),
            "text": final_text
        }

        processed_list.append(new_instance)

    return processed_list


def save_json_data(data: List[Dict[str, Any]], filename: str):
    """保存 JSON 格式的数据到指定文件"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # 使用 ensure_ascii=False 以便正确保存中文等非ASCII字符
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 成功保存 {len(data)} 条矛盾数据到 {filename}")
    except Exception as e:
        print(f"🚨 错误：保存文件 {filename} 失败: {e}")


def main():
    # 1. 加载和采样数据
    sampled_raw_data = load_and_sample_data(INPUT_FILE, SAMPLE_SIZE)

    if not sampled_raw_data:
        return

    # 2. 构造矛盾文本
    final_contradiction_data = construct_contradiction_text(sampled_raw_data)

    # 3. 保存结果
    save_json_data(final_contradiction_data, OUTPUT_FILE)

    # 打印预览
    print("\n[构造的矛盾文本预览 (第一个实例)]:")
    if final_contradiction_data:
        print(json.dumps(final_contradiction_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()