import json
import os
from typing import List, Dict, Any

# 输入和输出文件名/路径
INPUT_FILENAME = "../dense_text.json"
OUTPUT_FILENAME = "dense_text.txt" # 更改了输出文件名以区分和避免覆盖

def extract_and_flatten_contradictions_from_file(input_file: str, output_file: str):
    """
    读取指定 JSON 文件，提取每个文档中的矛盾句对，并将其扁平化保存到新的 JSON 文件。

    新增功能：在输出记录中包含原始文档的 'scope' 字段。
    """

    # --- 1. 从文件加载数据 ---
    print(f"尝试从文件加载数据: {input_file}...")

    if not os.path.exists(input_file):
        print(f"🔴 错误：未找到输入文件 {input_file}。请检查路径是否正确。")
        return

    input_data_list = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 假设文件内容是包含多个文档对象的 JSON 数组
            input_data_list = json.load(f)
        print(f"✅ 成功读取文件: {input_file}，共包含 {len(input_data_list)} 份文档。")

    except json.JSONDecodeError:
        print(f"🔴 错误：文件 {input_file} 不是有效的 JSON 格式。")
        return
    except Exception as e:
        print(f"🔴 错误：加载文件时发生异常: {e}")
        return

    # --- 2. 数据转换和扁平化 ---
    flattened_data: List[Dict[str, str]] = []
    total_pairs_extracted = 0

    for doc in input_data_list:
        source_id = doc.get("source_id", "N/A")
        doc_type = doc.get("doc_type", "Unknown")
        # --- 关键修改 1: 提取 scope 字段 ---
        scope = doc.get("scope", "N/A")

        pairs = doc.get("contradiction_pairs", [])

        for pair in pairs:
            text = pair.get('text', '')

            if '[SEP]' in text:
                try:
                    s1, s2 = [s.strip() for s in text.split('[SEP]', 1)]

                    new_item = {
                        "source_id": source_id,
                        "doc_type": doc_type,
                        # --- 关键修改 2: 添加 scope 字段到输出 ---
                        "scope": scope,
                        # ----------------------------------------
                        "sentence_1": s1,
                        "sentence_2": s2,
                        "contradiction_type": pair.get('contradiction_type', 'Unknown'),
                        "reason": pair.get('reason', 'No reason provided')
                    }
                    flattened_data.append(new_item)
                    total_pairs_extracted += 1

                except ValueError:
                    print(f"⚠️ 警告：跳过格式错误的矛盾句对 (ID: {source_id})")

    # --- 3. 保存为 JSON 文件 ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json_output = json.dumps(flattened_data, ensure_ascii=False, indent=4)
            f.write(json_output)

        print(f"\n✅ 成功提取并扁平化 {total_pairs_extracted} 条矛盾记录。")
        print(f"结果已保存到 {output_file} (JSON 数组格式)。新增字段: 'scope'。")

    except Exception as e:
        print(f"🔴 错误：保存文件时发生异常: {e}")


if __name__ == '__main__':
    extract_and_flatten_contradictions_from_file(INPUT_FILENAME, OUTPUT_FILENAME)