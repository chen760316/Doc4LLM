import json
import os
import random
from typing import Dict, Any, List, Set, Optional, Tuple

# --- 配置 ---
INPUT_FILE = "ContraDoc.json"
POS_SAMPLE_SIZE = 5
NEG_SAMPLE_SIZE = 20


# --- (保留 load_contra_doc_data 函数) ---

def load_contra_doc_data(file_path: str) -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """尝试从 ContraDoc.json 文件中加载数据。"""
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return {"pos": {}, "neg": {}}

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict) or 'pos' not in data or 'neg' not in data:
            print("🚨 错误：文件内容结构不符合预期的 ContraDoc 格式。")
            return None

        print(f"✅ 成功加载 ContraDoc 数据。")
        return data

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None


def sample_pos_data(pos_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """
    对 'pos' 条目进行基于多字段的非重复采样。
    返回采样字典 (以 ID 为键) 和已采样的 unique id 集合。
    """

    all_pos_docs = list(pos_data.items())
    # 🌟 修正：更改为字典，以 ID 为键
    sampled_pos_docs_dict: Dict[str, Dict[str, Any]] = {}
    sampled_ids: Set[str] = set()

    # 采样的字段和访问键
    sampling_criteria = [
        ("contra_type",
         lambda d: tuple(sorted(d["contra_type"]))
         if isinstance(d.get("contra_type"), list)
         else None),
        ("contra_plug", lambda d: d.get("contra_plug")),
        ("doc_type", lambda d: d.get("doc_type")),
        ("scope",
         lambda d: tuple(sorted(d["scope"]))
         if isinstance(d.get("scope"), list)
         else d.get("scope")),
    ]

    print("\n--- 📝 正在对 'pos' 数据进行非重复采样... ---")

    for field_name, key_func in sampling_criteria:
        grouped_docs: Dict[Any, List[Tuple[str, Dict[str, Any]]]] = {}
        for doc_id, doc_data in all_pos_docs:

            try:
                key = key_func(doc_data)
            except Exception:
                key = None

            if key is not None and doc_id not in sampled_ids:
                if key not in grouped_docs:
                    grouped_docs[key] = []
                grouped_docs[key].append((doc_id, doc_data))

        print(f"   - 字段 '{field_name}' 分组数: {len(grouped_docs)}")

        for key, available_docs in grouped_docs.items():
            num_to_sample = min(POS_SAMPLE_SIZE, len(available_docs))

            samples = random.sample(available_docs, num_to_sample)

            for doc_id, doc_data in samples:
                if doc_id not in sampled_ids:
                    # 🌟 修正：添加到字典中，键是 doc_id
                    sampled_pos_docs_dict[doc_id] = doc_data
                    sampled_ids.add(doc_id)

        print(f"     -> 采样后 'pos' 总数（不重复）: {len(sampled_ids)}")

    return sampled_pos_docs_dict, sampled_ids


def sample_neg_data(neg_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    对 'neg' 条目进行简单随机采样 20 个。
    返回采样字典 (以 ID 为键)。
    """
    print("\n--- 📝 正在对 'neg' 数据进行随机采样... ---")

    # 🌟 修正：保留键值对列表，而非只取值
    all_neg_docs_items = list(neg_data.items())

    num_available = len(all_neg_docs_items)
    num_to_sample = min(NEG_SAMPLE_SIZE, num_available)

    if num_available == 0:
        print("⚠️ 警告：'neg' 字典为空，无法采样。")
        return {}

    sampled_items = random.sample(all_neg_docs_items, num_to_sample)

    # 🌟 修正：将采样的键值对列表转换回字典
    sampled_neg_docs_dict = dict(sampled_items)

    print(f"   - 'neg' 总数: {num_available}，实际采样数: {len(sampled_neg_docs_dict)}")

    return sampled_neg_docs_dict


def main():
    # 1. 加载数据
    contra_doc_data = load_contra_doc_data(INPUT_FILE)
    if contra_doc_data is None:
        return

    # 2. 采样 'pos' 数据 (返回字典)
    pos_samples_dict, pos_ids = sample_pos_data(contra_doc_data.get('pos', {}))

    # 3. 采样 'neg' 数据 (返回字典)
    neg_samples_dict = sample_neg_data(contra_doc_data.get('neg', {}))

    # 4. 关键修正：将采样的字典结构保留在最终输出中
    final_output_structure: Dict[str, Dict[str, Dict[str, Any]]] = {
        "pos": pos_samples_dict,
        "neg": neg_samples_dict
    }

    # 5. 格式化输出 (将结果转换为 JSON 字符串)
    num_pos = len(pos_samples_dict)
    num_neg = len(neg_samples_dict)

    print("\n--- 🎉 采样结果统计 ---")
    print(f"   - 'pos' 最终采样数: {num_pos}")
    print(f"   - 'neg' 最终采样数: {num_neg}")
    print(f"   - 最终总采样数: {num_pos + num_neg}")

    # 转换为 JSON 格式字符串
    final_json_output = json.dumps(final_output_structure, indent=2, ensure_ascii=False)

    # 6. 打印/保存结果
    output_filename = "sampled_contradoc.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_json_output)

    print(f"\n✅ 采样数据已保存到: {output_filename}")
    print("\n[结构化采样数据预览]:")
    # 打印前几行以展示结构
    preview_lines = final_json_output.split('\n')
    # 打印到第一个 'neg' 或文件结束，以便展示 'pos' 的字典结构
    print('\n'.join(preview_lines[:min(len(preview_lines), 20)]) + "\n...")


if __name__ == "__main__":
    main()