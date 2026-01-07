"""
将原始的属性图json文本转为自然语言
暂时useless
"""

import json


def describe_entity(entity):
    """将节点属性转为自然语言描述"""
    # 排除一些不需要显示的元数据属性
    exclude_keys = {}

    # 获取所有需要显示的属性
    attrs = []
    for key, value in entity.items():
        if key not in exclude_keys and value not in [None, "", []]:
            # 处理列表类型的值
            if isinstance(value, list):
                if value:  # 只处理非空列表
                    attrs.append(f"{key.replace('_', ' ')}: {', '.join(map(str, value))}")
            else:
                attrs.append(f"{key.replace('_', ' ')}: {value}")

    name = entity.get("name", entity.get("label", "Unknown"))

    if attrs:
        return f"{name} ({'; '.join(attrs)})"
    return name


def describe_relation(entry):
    """将三元组转化为自然语言对"""
    # 提取节点和关系信息
    subj = describe_entity(entry.get("node_1", {}))
    obj = describe_entity(entry.get("node_2", {}))
    rel = entry.get("relationship", "related to")

    # 处理边属性
    edge_attrs = entry.get("edge_properties", {})
    if not edge_attrs and "properties" in entry:
        edge_attrs = entry.get("properties", {})

    # 构建关系描述
    if edge_attrs:
        edge_desc_items = []
        for k, v in edge_attrs.items():
            if v not in [None, "", []]:
                if isinstance(v, list):
                    edge_desc_items.append(f"{k.replace('_', ' ')}: {', '.join(map(str, v))}")
                else:
                    edge_desc_items.append(f"{k.replace('_', ' ')}: {v}")

        if edge_desc_items:
            full_sentence = f"{rel} ({'; '.join(edge_desc_items)})"
        else:
            full_sentence = rel
    else:
        full_sentence = rel

    return {
        "subject": subj,
        "relation": full_sentence,
        "object": obj,
        "subject_entity": entry.get("node_1", {}),
        "object_entity": entry.get("node_2", {}),
        "relationship_type": rel,
        "edge_properties": edge_attrs
    }


def process_knowledge_graph(input_file, output_file=None):
    """处理知识图谱数据"""
    # 加载JSON数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建自然语言句子对
    sentence_pairs = []
    for i, entry in enumerate(data):
        try:
            pair = describe_relation(entry)
            pair["id"] = i + 1
            sentence_pairs.append(pair)
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            continue

    # 输出示例
    print(f"Processed {len(sentence_pairs)} pairs")
    print("\n" + "=" * 60)

    for i, pair in enumerate(sentence_pairs[:3]):
        print(f"▶ Example {i + 1}")
        print(f"Subject: {pair['subject']}")
        print(f"Relation: {pair['relation']}")
        print(f"Object: {pair['object']}")
        print("-" * 50)

    # 保存结果
    if output_file:
        if output_file.endswith('.jsonl'):
            with open(output_file, "w", encoding="utf-8") as f:
                for item in sentence_pairs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(sentence_pairs, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {output_file}")

    return sentence_pairs


def print_jsonl_file(file_path):
    """打印JSONL文件的内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"📄 Contents of '{file_path}':")
            print("=" * 80)

            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    print(f"🔹 Entry {i}:")
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                    print("-" * 60)
                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing line {i}: {e}")
                    print(f"Line content: {line}")
                    print("-" * 60)

    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")


# 使用示例
if __name__ == "__main__":
    # 处理数据
    results = process_knowledge_graph(
        input_file="output_knowledge_graph.json",
        output_file="kg_sentence_pairs.jsonl"
    )
    # 打印文件内容
    # print_jsonl_file("kg_sentence_pairs.jsonl")