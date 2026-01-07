import json
from typing import Dict, Any, List, Set, Tuple

# --- 配置 ---
INPUT_FILE = "output_knowledge_graph_v3.json"
OUTPUT_FILE = "filtered_knowledge_graph.json"

def filter_single_occurrence_edges(input_path: str, output_path: str) -> None:
    """
    加载知识图谱边数据，过滤掉 node_1 和 node_2 实体名在整个图谱中都只出现一次的边。
    如果一条边被删除，则整个 { ... } 对象被删除。
    """
    try:
        # 1. 加载数据
        with open(input_path, 'r', encoding='utf-8') as f:
            edges: List[Dict[str, Any]] = json.load(f)

    except FileNotFoundError:
        print(f"🚨 错误：输入文件未找到: {input_path}")
        return
    except json.JSONDecodeError:
        print(f"🚨 错误：JSON 解析失败，请检查文件格式: {input_path}")
        return

    print(f"--- 🚀 开始过滤。原始边数: {len(edges)} ---")

    # 2. 统计所有节点名称的出现次数
    node_counts: Dict[str, int] = {}

    for edge in edges:
        # 统计 node_1
        name1 = edge.get("node_1", {}).get("name")
        if name1:
            node_counts[name1] = node_counts.get(name1, 0) + 1

        # 统计 node_2
        name2 = edge.get("node_2", {}).get("name")
        # 即使 name2 等于 name1，也应该独立统计 node_2 的出现，以反映它作为关系的第二个节点的角色
        # 传统的 KG 过滤中，我们只关心节点作为整体在图中的总出现次数。
        if name2:
            node_counts[name2] = node_counts.get(name2, 0) + 1

    # 3. 识别只出现一次的节点名称（单例节点）
    single_occurrence_nodes: Set[str] = {
        name for name, count in node_counts.items() if count == 1
    }

    print(f"   - 发现总节点数 (去重): {len(node_counts)}")
    print(f"   - 发现单例节点数 (只出现一次): {len(single_occurrence_nodes)}")

    # 4. 过滤边
    filtered_edges: List[Dict[str, Any]] = []
    dropped_count = 0

    for edge in edges:
        name1 = edge.get("node_1", {}).get("name")
        name2 = edge.get("node_2", {}).get("name")

        # 检查两个节点的名字是否都在单例节点集合中
        is_node1_single = (name1 is not None) and (name1 in single_occurrence_nodes)
        is_node2_single = (name2 is not None) and (name2 in single_occurrence_nodes)

        # 只有当两个节点都是单例时才删除整条边
        if is_node1_single and is_node2_single:
            dropped_count += 1
            pass  # 跳过该边，从而删除整个对象
        else:
            filtered_edges.append(edge)

    # 5. 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_edges, f, ensure_ascii=False, indent=2)

    print(f"--- 🎉 过滤完成 ---")
    print(f"   - 移除的边数 (删除的 JSON 对象数): {dropped_count}")
    print(f"   - 过滤后剩余边数: {len(filtered_edges)}")
    print(f"✅ 过滤后的数据已保存到: {output_path}")

# 只有在至少两个示例中node_1和node_2同时出现的例子才进行保留，别的例子全部删除
# def filter_non_repeated_node_pairs(input_path: str, output_path: str) -> None:
#     """
#     加载知识图谱边数据，过滤掉所有 node_1 和 node_2 构成的实体对
#     (node_1.name, node_2.name) 只在图中出现一次的边。
#
#     只有当 (node_1.name, node_2.name) 对在整个图谱中至少出现两次时，
#     涉及该对的所有边才被保留。
#     """
#     try:
#         # 1. 加载数据
#         with open(input_path, 'r', encoding='utf-8') as f:
#             edges: List[Dict[str, Any]] = json.load(f)
#
#     except FileNotFoundError:
#         print(f"🚨 错误：输入文件未找到: {input_path}")
#         return
#     except json.JSONDecodeError:
#         print(f"🚨 错误：JSON 解析失败，请检查文件格式: {input_path}")
#         return
#
#     print(f"--- 🚀 开始过滤 (基于重复的实体对)。原始边数: {len(edges)} ---")
#
#     # 2. 统计所有 (node_1.name, node_2.name) 对的出现次数
#     # Key: (name1, name2) Tuple
#     # Value: 出现次数
#     pair_counts: Dict[Tuple[str, str], int] = {}
#
#     # 存储所有边的 name 对，用于后续的查找，避免重复提取
#     all_pairs: List[Tuple[str, str]] = []
#
#     for edge in edges:
#         name1 = edge.get("node_1", {}).get("name")
#         name2 = edge.get("node_2", {}).get("name")
#
#         # 只有当两个名称都存在时才计入统计
#         if name1 and name2:
#             current_pair = (name1, name2)
#             pair_counts[current_pair] = pair_counts.get(current_pair, 0) + 1
#             all_pairs.append(current_pair)
#         else:
#             # 如果缺少 name，用一个特殊值标记，确保 all_pairs 的长度与 edges 保持一致
#             all_pairs.append(("", ""))
#
#             # 3. 识别重复出现的对 (出现次数 >= 2)
#     repeated_pairs: Set[Tuple[str, str]] = {
#         pair for pair, count in pair_counts.items() if count >= 2
#     }
#
#     print(f"   - 发现唯一实体对总数: {len(pair_counts)}")
#     print(f"   - 发现重复实体对 (出现 >= 2 次) 数: {len(repeated_pairs)}")
#
#     # 4. 过滤边：只有当边的 (name1, name2) 对在 repeated_pairs 集合中才保留
#     filtered_edges: List[Dict[str, Any]] = []
#     dropped_count = 0
#
#     for i, edge in enumerate(edges):
#         current_pair = all_pairs[i]
#
#         # 检查当前这对 (name1, name2) 是否在重复出现的集合中
#         if current_pair in repeated_pairs:
#             filtered_edges.append(edge)
#         else:
#             dropped_count += 1
#             pass  # 过滤掉该边
#
#     # 5. 保存结果
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(filtered_edges, f, ensure_ascii=False, indent=2)
#
#     print(f"--- 🎉 过滤完成 ---")
#     print(f"   - 移除的边数 (删除的 JSON 对象数): {dropped_count}")
#     print(f"   - 过滤后剩余边数: {len(filtered_edges)}")
#     print(f"✅ 过滤后的数据已保存到: {output_path}")

def count_total_edges(input_path: str) -> None:
    """
    统计 JSON 文件中边的总数量（即不去重统计）。
    这个数字等于 node_1、node_2 和 relationship 字段作为实例出现的总次数。
    """
    try:
        # 1. 加载数据
        with open(input_path, 'r', encoding='utf-8') as f:
            edges: List[Dict[str, Any]] = json.load(f)

    except FileNotFoundError:
        print(f"🚨 错误：输入文件未找到: {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败，请检查文件格式。错误信息: {e}")
        return

    # 2. 计算总边数
    total_edges = len(edges)

    print(f"--- 统计结果 ---")
    print(f"✅ 文件中的总边数 (JSON 对象数量): {total_edges}")
    print(f"因此，不去重统计，以下字段作为实例的总次数均为：{total_edges}")
    print(f"   - 'node_1' 实例总数: {total_edges}")
    print(f"   - 'node_2' 实例总数: {total_edges}")
    print(f"   - 'relationship' 实例总数: {total_edges}")

    # 3. 额外验证：随机检查一个对象以确认结构
    if total_edges > 0:
        first_edge = edges[0]
        has_node1 = 'node_1' in first_edge
        has_node2 = 'node_2' in first_edge
        has_rel = 'relationship' in first_edge

        print("\n[结构检查]:")
        print(f"   - 示例对象包含 'node_1': {has_node1}")
        print(f"   - 示例对象包含 'node_2': {has_node2}")
        print(f"   - 示例对象包含 'relationship': {has_rel}")

        if has_node1 and has_node2 and has_rel:
            print("结论：JSON 结构正确，三者总数量相等。")
        else:
            print("🚨 警告：部分对象缺少关键字段，请检查 LLM 输出。")

# --- 主程序执行 ---
if __name__ == "__main__":
    filter_single_occurrence_edges(INPUT_FILE, OUTPUT_FILE)
    count_total_edges(OUTPUT_FILE)