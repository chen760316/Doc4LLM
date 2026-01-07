import json
import os
from typing import Dict, Any, List


# --- 文件加载函数 ---

def load_json_from_directory(directory: str) -> List[Dict[str, Any]]:
    """读取待补全属性的json文件路径（双星结构）"""
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_data = []
    for file in json_files:
        file_path = os.path.join(directory, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # dual_star_json 文件中的每个文件通常包含一个字典，其中包含 'stars' 和 'virtual_links'
                json_data.append(json.load(f))
        except Exception as e:
            print(f"[ERROR] 无法加载文件 {file_path}: {e}")
            continue
    return json_data


def load_source_graph_data(source_graph_path: str) -> List[Dict[str, Any]]:
    """加载用于属性补全的知识图谱文件 (filtered_knowledge_graph_enrich.json)"""
    try:
        with open(source_graph_path, 'r', encoding='utf-8') as f:
            source_graph = json.load(f)
        if not isinstance(source_graph, list):
            print(f"[ERROR] 源图文件内容不是列表。")
            return []
        return source_graph
    except Exception as e:
        print(f"[FATAL] 加载源图文件失败 {source_graph_path}: {e}")
        return []


# --- 属性补全核心函数 ---

def fill_node_and_edge_attributes(data: List[Dict[str, Any]], source_graph: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """根据提供的源图补全节点和边的属性"""

    # 1. 创建字典来存储所有节点和边的属性，按 name/relationship 作为索引
    node_attrs = {}
    edge_attrs = {}

    # 将源图中的所有节点和边属性存储在字典中
    for entry in source_graph:
        # 提取节点属性
        node_1 = entry.get("node_1", {})
        node_2 = entry.get("node_2", {})

        # 节点属性（使用名称作为键，确保是字典且有名称）
        if isinstance(node_1, dict) and "name" in node_1:
            node_attrs[node_1["name"]] = node_1
        if isinstance(node_2, dict) and "name" in node_2:
            node_attrs[node_2["name"]] = node_2

        # 边属性
        relationship = entry.get("relationship")
        edge_properties = entry.get("edge_properties", {})
        if relationship and edge_properties:
            # 使用关系作为键，并合并属性（如果有重复关系描述，则后续属性会覆盖前面的）
            if relationship not in edge_attrs:
                edge_attrs[relationship] = {}
            edge_attrs[relationship].update(edge_properties)

    # 2. 填充双星数据
    for file_data in data:  # file_data 是一个双星字典，包含 'stars' 键
        if not isinstance(file_data, dict) or "stars" not in file_data:
            print(f"[WARN] 忽略数据项：不是有效的双星结构字典。")
            continue

        # 双星结构包含两个星型结构 (star1 和 star2)
        for entry in file_data.get("stars", []):
            if not isinstance(entry, dict) or "center" not in entry:
                continue

            center = entry.get("center")
            if center is None:
                continue

            # 移除可能存在的后缀 (_1, _2) 以匹配源图的原始名称
            # 这是一个关键步骤，因为双星结构中的 center 可能带后缀
            center_key = center.rsplit('_', 1)[0] if center.endswith(('_1', '_2')) else center

            neighbors = entry.get("neighbors", [])
            multi_hop_neighbors = entry.get("multi_hop_neighbors", [])

            # 初始化属性字典
            entry.setdefault("node_attrs", {})
            entry.setdefault("edges_attrs", {})
            entry.setdefault("multi_hop_node_attrs", {})
            entry.setdefault("multi_hop_edges_attrs", {})

            # 2.1 补全 center 的节点属性 (使用 center_key)
            if center_key in node_attrs:
                entry["node_attrs"][center] = node_attrs[center_key]

            # 2.2 补全 neighbors (单跳邻居) 的节点属性
            for node in neighbors:
                node_key = node.rsplit('_', 1)[0] if node.endswith(('_1', '_2')) else node
                if node_key in node_attrs:
                    entry["node_attrs"][node] = node_attrs[node_key]

            # 2.3 补全 edges (单跳边) 的边属性
            for edge_description in entry.get("edges", []):
                if edge_description in edge_attrs:
                    entry["edges_attrs"][edge_description] = edge_attrs[edge_description]

            # 2.4 补全 multi_hop_neighbors (多跳邻居) 的节点属性和边属性
            for hop in multi_hop_neighbors:
                hop_node = hop.get("neighbor")
                hop_edge_description = hop.get("edge")

                # 节点属性
                if hop_node:
                    hop_node_key = hop_node.rsplit('_', 1)[0] if hop_node.endswith(('_1', '_2')) else hop_node
                    if hop_node_key in node_attrs:
                        entry["multi_hop_node_attrs"][hop_node] = node_attrs[hop_node_key]

                # 边属性
                if hop_edge_description and hop_edge_description in edge_attrs:
                    entry["multi_hop_edges_attrs"][hop_edge_description] = edge_attrs[hop_edge_description]

    return data


# --- 文件保存函数 ---

def save_filled_json(filled_data: List[Dict[str, Any]], output_directory: str):
    """保存补全后的json文件到指定目录"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 尝试使用原始文件名保存，如果没有原始文件名，则使用索引
    for i, data in enumerate(filled_data):
        # 由于 load_json_from_directory 丢失了文件名，这里只能使用索引
        output_file = os.path.join(output_directory, f"filled_dual_star_{i + 1}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ Saved filled JSON to: {output_file}")


# --- 主执行函数 ---

def main():
    print("--- 🚀 开始属性补全 (使用单个源图) ---")

    # 文件路径设置
    input_directory = "dual_star_json"  # 输入目录路径
    # 🎯 修正点 1: 单个源图路径
    source_graph_path = "filtered_dense_property_graph_enrich.json"
    output_directory = "dual_star_json_with_attrs"  # 输出目录路径

    # 读取待补全数据 (双星结构)
    data = load_json_from_directory(input_directory)
    print(f"[INFO] 成功加载 {len(data)} 个双星结构文件。")

    # 读取源图数据
    source_graph = load_source_graph_data(source_graph_path)
    if not source_graph:
        return

    # 🎯 修正点 2: 补全节点和边的属性 (只传入一个源图)
    filled_data = fill_node_and_edge_attributes(data, source_graph)

    # 保存补全后的数据
    save_filled_json(filled_data, output_directory)

    print("\n--- ✅ 任务完成 ---")


if __name__ == "__main__":
    main()