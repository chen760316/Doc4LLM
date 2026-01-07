"""
将原始属性图json文本中顶点和边的属性转为自然语言
方便顶点和边的属性对进行两两比较
暂时useless
"""

import json


def describe_entity_attributes(entity, entity_type):
    """生成节点属性的自然语言描述列表"""
    if not entity:
        return []

    descriptions = []
    entity_name = entity.get("name", entity.get("label", entity_type))

    for key, value in entity.items():
        if value not in [None, "", []]:
            key_name = key.replace('_', ' ')

            if isinstance(value, list):
                if value:
                    items = ', '.join(map(str, value))
                    descriptions.append(f"{entity_name} has {key_name}: {items}")
            else:
                descriptions.append(f"{entity_name}'s {key_name} is {value}")

    return descriptions


def describe_relationship_with_attributes(relationship, edge_attrs):
    """生成包含关系类型和属性的完整自然语言描述"""
    if not edge_attrs:
        return [f"relationship type: {relationship}"]

    descriptions = [f"relationship type: {relationship}"]
    for k, v in edge_attrs.items():
        if v not in [None, "", []]:
            key_name = k.replace('_', ' ')
            if isinstance(v, list):
                if v:
                    items = ', '.join(map(str, v))
                    descriptions.append(f"relationship has {key_name}: {items}")
            else:
                descriptions.append(f"relationship's {key_name} is {v}")

    return descriptions


def create_minimal_output(entry):
    """创建只包含自然语言描述的简化输出"""
    subject_entity = entry.get("node_1", {})
    object_entity = entry.get("node_2", {})
    rel = entry.get("relationship", "related to")

    # 处理边属性
    edge_attrs = entry.get("edge_properties", {})
    if not edge_attrs and "properties" in entry:
        edge_attrs = entry.get("properties", {})

    # 生成自然语言描述
    subject_attrs = describe_entity_attributes(subject_entity, "subject")
    object_attrs = describe_entity_attributes(object_entity, "object")
    relationship_desc = describe_relationship_with_attributes(rel, edge_attrs)

    return {
        "id": entry.get("id"),
        "subject_attributes": subject_attrs,
        "object_attributes": object_attrs,
        "relationship": relationship_desc
    }


def process_knowledge_graph_minimal(input_file, output_file=None, example_output_file="example_output.txt"):
    """处理知识图谱数据，只保存自然语言描述"""
    # 加载JSON数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建自然语言描述
    minimal_pairs = []
    for i, entry in enumerate(data):
        try:
            minimal_entry = create_minimal_output(entry)
            minimal_entry["id"] = i + 1
            minimal_pairs.append(minimal_entry)
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            continue

    # 输出示例
    # print(f"Processed {len(minimal_pairs)} pairs")
    # print("\n" + "=" * 60)
    #
    # for i, pair in enumerate(minimal_pairs[:2]):
    #     print(f"▶ Example {i + 1}")
    #     print("Subject attributes:")
    #     for attr in pair['subject_attributes']:
    #         print(f"  - {attr}")
    #
    #     print("Object attributes:")
    #     for attr in pair['object_attributes']:
    #         print(f"  - {attr}")
    #
    #     print("Relationship:")
    #     for attr in pair['relationship']:
    #         print(f"  - {attr}")
    #
    #     print("-" * 60)

    # 保存结果
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in minimal_pairs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if example_output_file:
        with open(example_output_file, "w", encoding="utf-8") as f:
            for i, item in enumerate(minimal_pairs):
                f.write(f"▶ Entry {i + 1}\n")
                f.write("Subject attributes:\n")
                for attr in item['subject_attributes']:
                    f.write(f"  - {attr}\n")
                f.write("Object attributes:\n")
                for attr in item['object_attributes']:
                    f.write(f"  - {attr}\n")
                f.write("Relationship:\n")
                for attr in item['relationship']:
                    f.write(f"  - {attr}\n")
                f.write("-" * 60 + "\n")

        print(f"Minimal results saved to {example_output_file}")

    return minimal_pairs


def print_minimal_jsonl(file_path, max_entries=5):
    """打印简化版JSONL文件的内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"📄 Contents of '{file_path}':")
            print("=" * 80)

            for i, line in enumerate(f, 1):
                if i > max_entries:
                    print(f"... and {max_entries} more entries")
                    break

                data = json.loads(line.strip())
                print(f"🔹 Entry {data['id']}:")

                if data['subject_attributes']:
                    print("Subject attributes:")
                    for attr in data['subject_attributes']:
                        print(f"  • {attr}")

                if data['object_attributes']:
                    print("Object attributes:")
                    for attr in data['object_attributes']:
                        print(f"  • {attr}")

                if data['relationship']:
                    print("Relationship:")
                    for attr in data['relationship']:
                        print(f"  • {attr}")

                print("-" * 60)

    except FileNotFoundError:
        print(f"❌ File '{file_path}' not found.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")


# 使用示例
if __name__ == "__main__":
    # 处理数据并生成最小化输出
    minimal_results = process_knowledge_graph_minimal(
        input_file="output_knowledge_graph.json",
        output_file="kg_minimal_pairs.jsonl"
    )

    # 打印文件内容
    print_minimal_jsonl("kg_minimal_pairs.jsonl")