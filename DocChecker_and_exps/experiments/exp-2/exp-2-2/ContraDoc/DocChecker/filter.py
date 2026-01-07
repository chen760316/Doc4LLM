import json
import os
import re
from typing import List, Dict, Any, Set


# --- 辅助函数：深度规范化 ---
def normalize_text(text: str) -> str:
    """
    深度规范化文本：移除首尾空白，并将所有连续的空白字符替换为单个空格。
    """
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- 核心修改函数 ---
def load_and_normalize_exclusion_keys(file_path: str) -> Set[str]:
    """
    加载 result.txt 文件，去除 'Sentence 1:' / 'Sentence 2:' 前缀，规范化句子，
    并生成 [S1] [SEP] [S2] 格式的键，包含双向匹配。
    """
    exclusion_set = set()
    s1_prefix = "Sentence 1:"
    s2_prefix = "Sentence 2:"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("句子对:"):
                    pair_str = line.split("句子对:", 1)[1].strip()

                    if '/' in pair_str:
                        # 原始带前缀的 S1 和 S2
                        raw_s1, raw_s2 = pair_str.split('/', 1)

                        # 1. 去除前缀和首尾空格
                        # 对于 S1
                        s1 = raw_s1.strip()
                        if s1.startswith(s1_prefix):
                            s1 = s1[len(s1_prefix):].strip()

                        # 对于 S2
                        s2 = raw_s2.strip()
                        if s2.startswith(s2_prefix):
                            s2 = s2[len(s2_prefix):].strip()

                        # 2. 深度规范化处理
                        s1 = normalize_text(s1)
                        s2 = normalize_text(s2)

                        # 3. 规范化键：[S1] [SEP] [S2]
                        normalized_key = f"{s1} [SEP] {s2}"
                        exclusion_set.add(normalized_key)

                        # 4. 反向键：[S2] [SEP] [S1]
                        reverse_key = f"{s2} [SEP] {s1}"
                        exclusion_set.add(reverse_key)

    except Exception as e:
        print(f"警告: 处理排除文件时出错: {e}")

    return exclusion_set


# --- 保持其他函数不变 (load_dense_data 和 exclude_examples_by_strict_match) ---

def load_dense_data(file_path: str) -> List[Dict[str, Any]]:
    # 此函数与前一个回答中更新的版本保持一致
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'text' in item:
                    item['text'] = normalize_text(item['text'])
            return data
    except Exception as e:
        print(f"错误: 文件 {file_path} 处理失败: {e}")
        return []


def exclude_examples_by_strict_match(dense_data: List[Dict[str, Any]], exclusion_keys: Set[str]) -> List[Dict[str, Any]]:
    # 此函数与前一个回答中更新的版本保持一致
    filtered_data = []
    excluded_count = 0

    for item in dense_data:
        dense_text = item.get('text', '')
        if dense_text in exclusion_keys:
            excluded_count += 1
        else:
            filtered_data.append(item)

    print(f"\n--- 筛选结果 (深度规范化后的严格匹配) ---")
    print(f"原始数据总数: {len(dense_data)}")
    print(f"排除的示例数: {excluded_count}")
    print(f"剩余的示例数: {len(filtered_data)}")

    return filtered_data


# --- 主执行部分 ---
if __name__ == "__main__":
    DENSE_FILE_PATH = "dense_text.json"
    EXCLUSION_FILE_PATH = "result.txt"
    OUTPUT_FILE_PATH = "filtered_dense_text.json"

    # 1. 加载 dense_text.json (已包含规范化)
    dense_data = load_dense_data(DENSE_FILE_PATH)
    if not dense_data:
        exit()

    # 2. 加载并规范化 result.txt 中的排除键 (已去除 'Sentence 1:'/'Sentence 2:' 前缀)
    exclusion_keys = load_and_normalize_exclusion_keys(EXCLUSION_FILE_PATH)

    # 3. 执行排除操作
    filtered_data = exclude_examples_by_strict_match(dense_data, exclusion_keys)

    # 4. 保存筛选后的结果
    try:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 筛选后的数据已保存到: {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")