import json
import os
from typing import List, Dict, Any

# 假设的输入文件名
INPUT_FILENAME = "dense_text.json"
# 输出文件名
OUTPUT_FILENAME = "dense_text.txt"


def transform_and_save_data(input_file: str, output_file: str):
    """
    读取指定输入文件 (INPUT_FILENAME)，转换数据结构，并保存为 TXT 文件。

    关键修改: 输出文件最外层使用 [] 包裹，使其成为一个有效的 JSON 数组。
    """

    # --- 1. 加载输入数据 ---
    input_data = []

    if not os.path.exists(input_file):
        # 如果文件不存在，则使用模拟数据并发出警告
        print(f"🔴 错误：未找到输入文件 {input_file}。使用模拟数据进行演示。")
        input_data = [
            {
                "contradiction_type": "Causal",
                "reason": "The first sentence states that the delay is not due to Lynch being African-American, while the second sentence suggests that race is a major factor in the delay.",
                "text": "Senate Republicans adamantly deny the delay in scheduling a vote on Lynch's nomination is because she is African-American. [SEP] I think race certainly can be considered as a major factor in the reason for this delay, but it's also the irrationality of the new Republicans.",
            },
            {
                "contradiction_type": "Numeric",
                "reason": "The first sentence implies a specific numeric comparison with the eight previous nominees, while the second sentence suggests a broader comparison to 'any attorney general nominee in recent history,' which could contradict the numeric claim.",
                "text": "By the time the Senate returns from Easter recess on Monday, it'll have been longer than the eight previous nominees for the job -- combined. [SEP] The period between the Senate Judiciary Committee's vote to confirm and the full Senate vote -- which in Lynch's case has not been scheduled -- has lasted longer for her than for any attorney general nominee in recent history.",
            }
        ]
    else:
        # 如果文件存在，则尝试读取
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            print(f"✅ 成功读取文件: {input_file}")

        except json.JSONDecodeError:
            print(f"🔴 错误：文件 {input_file} 不是有效的 JSON 格式。")
            return
        except Exception as e:
            print(f"🔴 错误：加载文件时发生异常: {e}")
            return

    # --- 2. 数据转换 ---
    transformed_data: List[Dict[str, str]] = []

    for item in input_data:
        if 'text' in item and isinstance(item['text'], str) and '[SEP]' in item['text']:
            try:
                # 按 [SEP] 分割，并去除句子前后的空白
                s1, s2 = [s.strip() for s in item['text'].split('[SEP]', 1)]

                new_item = {
                    "sentence_1": s1,
                    "sentence_2": s2,
                    "contradiction_type": item.get('contradiction_type', 'Unknown'),
                    "reason": item.get('reason', 'No reason provided')
                }
                transformed_data.append(new_item)

            except ValueError:
                print(f"⚠️ 警告：跳过格式错误的条目 (Text: {item.get('text', 'N/A')})")

    # --- 3. 保存为 .txt 文件 (最外层使用 [] 包裹) ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 将整个列表对象转换为 JSON 字符串，并使用 indent=4 使其格式美观
            # 这一步会自动在最外层添加 []
            json_output = json.dumps(transformed_data, ensure_ascii=False, indent=4)
            f.write(json_output)

        print(f"\n✅ 成功转换 {len(transformed_data)} 条记录并保存到 {output_file}")
        print(f"   注意：输出文件 {output_file} 现在是一个包含所有 JSON 对象的 JSON 数组 ([] 包裹)。")

    except Exception as e:
        print(f"🔴 错误：保存文件时发生异常: {e}")


if __name__ == '__main__':
    transform_and_save_data(INPUT_FILENAME, OUTPUT_FILENAME)