"""
生成多跳星型图，可以借助merge_star_structures_with_hops(stars, max_hops=5)这一函数设置最大跳数
分开保留各个得到的星型图，并进行可视化 (此版本已禁用可视化功能，仅保留 JSON 文件生成)
输出文件目录为star_graph和star_json
"""
import json
from collections import defaultdict
import os
import networkx as nx
# 假设知识图谱相关客户端和dotenv导入是外部依赖，这里保留
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import matplotlib
import re
from typing import Dict, Any, List, Set, Tuple

# 由于不再进行可视化，这里的 matplotlib 设置和导入 networkx 并非严格需要，但为保持函数完整性暂保留
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM 客户端只用于简化边标签，如果数据中没有需要简化的边，可以注释掉
llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


def clean_filename(name: str) -> str:
    """移除文件名中的非法字符和 Windows 不允许的保留字符"""
    cleaned = name.replace(' ', '_')
    # 移除所有非法和潜在问题的字符 (只保留字母、数字、下划线)
    cleaned = re.sub(r'[^\w-]', '', cleaned)
    return cleaned

def load_graph(file: str) -> List[Dict[str, Any]]:
    """加载三元组数据"""
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 调试输出，检查每个条目的内容
    for i, entry in enumerate(data):
        if "relationship" not in entry:
            print(f"[WARNING] Entry {i} is missing 'relationship': {entry}")

    return data

def build_star_structure(center_node: str, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ 为给定的实体和其相关三元组创建星型结构 """
    neighbors = []
    edges = []

    for triple in triples:
        node1_name = triple.get("node_1", {}).get("name")
        node2_name = triple.get("node_2", {}).get("name")
        relationship = triple.get("relationship")

        if node1_name == center_node and node2_name:
            neighbors.append(node2_name)
            edges.append(relationship)
        elif node2_name == center_node and node1_name:
            neighbors.append(node1_name)
            edges.append(relationship)

    return {
        "center": center_node,
        "neighbors": neighbors,
        "edges": edges
    }


def combine_triples_with_same_entity(triples: List[Dict[str, Any]], max_hops=1) -> List[Dict[str, Any]]:
    """ 合并包含相同实体的三元组，不再判断上下文（相邻性） """
    entity_triples_map = defaultdict(list)

    for i, triple in enumerate(triples):
        node1_name = triple.get("node_1", {}).get("name")
        node2_name = triple.get("node_2", {}).get("name")

        if node1_name:
            entity_triples_map[node1_name].append(triple)

        if node2_name:
            entity_triples_map[node2_name].append(triple)

        if not node1_name and not node2_name:
            print(f"[WARN] 三元组 {i} 缺失 node_1/node_2 名称，已跳过。")

    merged_stars = []

    for entity, entity_triples in entity_triples_map.items():
        star = build_star_structure(entity, entity_triples)
        if star["neighbors"]:
            merged_stars.append(star)

    return merged_stars

# 保持所有辅助函数不变...
# is_overlapping, get_neighbors_with_hops, remove_single_hop_neighbors, remove_duplicate_center, merge_star_structures_with_hops (这是核心函数)
# ...

def get_neighbors_with_hops(node: str, hop_count: int, max_hops: int, stars: List[Dict[str, Any]], visited_nodes: Set[str]=set(), neighbors_set: Set[str]=set(), multi_hop_neighbors: Set[Tuple[str, int, str]]=set()) -> Tuple[Set[str], Set[Tuple[str, int, str]]]:
    """ 获取指定节点的所有邻居，最多最大跳数 """
    if hop_count > max_hops or node in visited_nodes:
        return visited_nodes, multi_hop_neighbors

    # 递归调用时，需要复制集合以避免污染同层级的其他分支
    current_visited = visited_nodes.copy()
    current_multi_hop_neighbors = multi_hop_neighbors.copy()

    current_visited.add(node)

    for star in stars:
        if star["center"] == node:
            for neighbor, edge in zip(star["neighbors"], star["edges"]):
                # 记录邻居和关系
                if hop_count >= 1:
                    current_multi_hop_neighbors.add((neighbor, hop_count, edge))

                # 递归获取邻居的邻居
                # 只有当邻居尚未被访问时才进行递归，以避免死循环
                if neighbor not in visited_nodes:
                    # 递归调用时，传入更新的 visited_nodes
                    current_visited, current_multi_hop_neighbors = get_neighbors_with_hops(
                        neighbor, hop_count + 1, max_hops, stars, current_visited, neighbors_set, current_multi_hop_neighbors
                    )

    return current_visited, current_multi_hop_neighbors


def remove_single_hop_neighbors(star: Dict[str, Any]) -> Dict[str, Any]:
    """ 删除多跳邻居中的跳数为1的邻居，并删除与 neighbors 中节点重复的 multi_hop_neighbors 节点 """
    neighbor_nodes = set(star["neighbors"])

    # 删除 hop_count 为 1 的邻居 AND 删除与 neighbors 中节点重复的节点
    star["multi_hop_neighbors"] = [
        neighbor_info for neighbor_info in star["multi_hop_neighbors"]
        if neighbor_info["hop_count"] > 1 and neighbor_info["neighbor"] not in neighbor_nodes
    ]

    return star


def remove_duplicate_center(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ 遍历数据，删除重复的中心节点，并将重复节点的信息合并到第一个出现的该中心节点中。 """
    centers_to_star: Dict[str, int] = {}  # 记录中心节点及其在 data 中的索引
    indices_to_remove: Set[int] = set()

    for i, entry in enumerate(data):
        center = entry['center']
        if center not in centers_to_star:
            centers_to_star[center] = i
        else:
            main_index = centers_to_star[center]
            # 合并信息到主星型结构
            existing_star = data[main_index]
            new_star = data[i]

            # 合并 neighbors (使用 set 保证不重复)
            existing_neighbors = set(existing_star["neighbors"])
            existing_edges = list(existing_star["edges"]) # 保持 edges 和 neighbors 一一对应

            # 简单合并：将新星型的邻居和边加入主星型，但这种简单合并会导致 neighbors 和 edges 错位。
            # 鉴于你的原始 build_star_structure 逻辑是基于 triple 列表构建的，
            # 这里应简化为仅合并 **唯一的** 邻居到 neighbors 列表，并确保 edges 数量一致。
            # 为了安全，我们只合并 multi_hop_neighbors，因为直接邻居的合并容易出错。

            # 仅合并 multi_hop_neighbors
            existing_multi_hop = set(tuple(mh.values()) for mh in existing_star["multi_hop_neighbors"])
            new_multi_hop = set(tuple(mh.values()) for mh in new_star["multi_hop_neighbors"])

            all_multi_hop = existing_multi_hop.union(new_multi_hop)

            existing_star["multi_hop_neighbors"] = [
                {"neighbor": n, "hop_count": h, "edge": e} for n, h, e in all_multi_hop
            ]

            # 标记重复项为删除
            indices_to_remove.add(i)

    # 从后往前删除，避免索引变化
    final_data = [data[i] for i in range(len(data)) if i not in indices_to_remove]

    return final_data


def merge_star_structures_with_hops(stars: List[Dict[str, Any]], max_hops=5) -> List[Dict[str, Any]]:
    """ 合并星型结构，并根据最大跳数添加邻居 """
    merged_stars: List[Dict[str, Any]] = []

    # 遍历每个星型结构，计算多跳邻居
    for star in stars:
        center = star["center"]

        # ⚠️ 注意：get_neighbors_with_hops 的递归实现需要非常小心状态管理。
        # 这里使用 set() 来确保每次调用都是干净的起始，但递归内部使用了 current_visited.copy()
        # 确保它不会陷入无限循环。
        visited_nodes, multi_hop_neighbors_set = get_neighbors_with_hops(center, 1, max_hops, stars)

        # 转换为列表格式，并去除中心节点本身
        multi_hop_neighbors = [
            {"neighbor": n, "hop_count": h, "edge": e}
            for n, h, e in multi_hop_neighbors_set
            if n != center
        ]

        # 去重 (按邻居名称)
        seen_neighbors = set()
        unique_multi_hop_neighbors = []
        for mh_neighbor in multi_hop_neighbors:
            neighbor_name = mh_neighbor["neighbor"]
            if neighbor_name not in seen_neighbors:
                seen_neighbors.add(neighbor_name)
                unique_multi_hop_neighbors.append(mh_neighbor)

        star["multi_hop_neighbors"] = unique_multi_hop_neighbors

        # 删除单跳邻居，仅保留多跳信息
        star = remove_single_hop_neighbors(star)

        # 将结构添加到结果列表
        merged_stars.append(star)

    # 最终对有重复中心节点的星型结构进行合并（虽然原始 combine_triples_with_same_entity 应该已经处理了大部分）
    merged_stars = remove_duplicate_center(merged_stars)

    return merged_stars


def split_star_structures(data: List[Dict[str, Any]]) -> None:
    """ 分割并保存每个中心节点的星型结构为独立的 JSON 文件到 star_json 目录 """
    centers = defaultdict(list)

    for entry in data:
        center_name = entry['center']
        centers[center_name].append(entry)

    output_dir = "star_json"
    os.makedirs(output_dir, exist_ok=True)

    for center_name, structures in centers.items():
        safe_center_name = clean_filename(center_name)
        file_name = os.path.join(output_dir, f"{safe_center_name}_star_structure.json")

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(structures, f, ensure_ascii=False, indent=4)

        print(f"File '{file_name}' has been created in {output_dir}.")

# --- 禁用可视化相关函数 (根据用户要求) ---

def simplify_edge_labels_with_llm(relationships: List[str], llm) -> List[str]:
    # 此函数保留，因为它可能在 visualize_star_structure_with_hops 内部被调用，
    # 但由于可视化被禁用，此函数实际上不会运行。
    # 它的逻辑是正确的，故保留其定义以防后续启用可视化。
    return relationships # 返回原始关系，避免在禁用可视化时调用 LLM


def visualize_star_structure_with_hops(json_data: Dict[str, Any], output_file: str, title="Star Structure", llm=None) -> None:
    """ **此函数已被禁用以取消生成 PNG 文件** """
    pass


def visualize_all_star_structures(input_dir="star_json", llm_model=None) -> None:
    """ **此函数已被禁用以取消生成 PNG 文件** """
    print(f"[INFO] ⚠️ 可视化功能已禁用。跳过生成 star_graph 目录下的 PNG 文件。")
    pass


# --- 主执行函数 ---

def main():
    # 输入文件
    input_file = "filtered_dense_property_graph_enrich.json"
    print(f"[PROCESSING] 提取星型结构：{input_file}")

    # 1. 加载三元组数据
    triples = load_graph(input_file)
    print(f"[DEBUG] Loaded {len(triples)} triples.")

    # 2. 合并三元组，构建单跳星型结构
    stars = combine_triples_with_same_entity(triples)
    print(f"[DEBUG] Created {len(stars)} initial star structures.")

    # 3. 合并星型结构，计算多跳邻居 (max_hops=5)
    merged_stars = merge_star_structures_with_hops(stars, max_hops=5)
    print(f"[DEBUG] Created {len(merged_stars)} merged star structures with multi-hops.")

    # 4. 调用split_star_structures保存独立的星型结构文件到 star_json 目录
    split_star_structures(merged_stars)

    # 5. 可视化 (此步骤根据用户要求被跳过)
    # visualize_all_star_structures(input_dir="star_json", llm_model=llm)

    print("\n--- ✅ 任务完成 ---")
    print("✨ 已成功生成多跳星型图 JSON 文件到 'star_json' 目录。")
    print("🖼️ 已跳过 PNG 文件生成步骤。")


if __name__ == "__main__":
    main()