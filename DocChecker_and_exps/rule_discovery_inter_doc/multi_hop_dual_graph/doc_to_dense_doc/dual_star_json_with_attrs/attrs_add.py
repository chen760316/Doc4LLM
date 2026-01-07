import json
import os

# 读取待补全属性的json文件路径
def load_json_from_directory(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_data = []
    for file in json_files:
        with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
            json_data.append(json.load(f))
    return json_data

# 加载graph1_enriched和graph2_enriched文件
def load_graph_data(graph1_path, graph2_path):
    with open(graph1_path, 'r', encoding='utf-8') as f:
        graph1 = json.load(f)
    with open(graph2_path, 'r', encoding='utf-8') as f:
        graph2 = json.load(f)
    return graph1, graph2

# 根据提供的graph1和graph2补全节点和边的属性
def fill_node_and_edge_attributes(data, graph1, graph2):
    # 创建一个字典来存储所有节点的属性，按 name 作为索引
    node_attrs_graph1 = {}
    node_attrs_graph2 = {}

    # 将 graph1 和 graph2 中的节点属性分别存储在字典中
    for graph in [graph1, graph2]:
        for entry in graph:
            node_1 = entry["node_1"]
            node_2 = entry["node_2"]
            if graph == graph1:
                node_attrs_graph1[node_1["name"]] = node_1
                node_attrs_graph1[node_2["name"]] = node_2
            elif graph == graph2:
                node_attrs_graph2[node_1["name"]] = node_1
                node_attrs_graph2[node_2["name"]] = node_2

    # 填充原始数据
    for file_data in data:  # 对每个文件进行处理
        if isinstance(file_data, list):  # 确保每个文件的内容是一个列表
            for entry in file_data:  # 对文件中的每一项进行处理
                if isinstance(entry, dict) and "center" in entry:
                    center = entry["center"]
                    if center is None:
                        continue

                    neighbors = entry.get("neighbors", [])
                    multi_hop_neighbors = entry.get("multi_hop_neighbors", [])

                    # 根据center判断是使用graph1还是graph2
                    if center == "Emily Clark":  # 第一个center是Emily Clark，使用graph1
                        node_attrs = node_attrs_graph1
                    else:  # 第二个center是Emily Clark，使用graph2
                        node_attrs = node_attrs_graph2

                    # 1. 补全 center 的节点属性
                    if center in node_attrs:
                        entry.setdefault("node_attrs", {}).setdefault(center, {}).update(node_attrs[center])

                    # 2. 补全 neighbors 中的节点属性
                    for node in neighbors:
                        if node in node_attrs:
                            node_attr = node_attrs[node]
                            entry.setdefault("node_attrs", {}).setdefault(node, {}).update(node_attr)

                    # 3. 补全 edges 中的边属性
                    for i, neighbor in enumerate(neighbors):
                        edge_description = entry["edges"][i]
                        # 生成完整的 edge 属性键：关系描述作为键
                        edge_key = edge_description
                        for graph in [graph1, graph2]:
                            for edge in graph:
                                if edge["relationship"] == edge_description:
                                    edge_props = edge.get("edge_properties", {})
                                    entry.setdefault("edges_attrs", {}).setdefault(edge_key, {}).update(edge_props)

                    # 4. 补全 multi_hop_neighbors 中的节点属性和边属性
                    for hop in multi_hop_neighbors:
                        hop_node = hop["neighbor"]
                        if hop_node in node_attrs:
                            node_attr = node_attrs[hop_node]
                            entry.setdefault("multi_hop_node_attrs", {}).setdefault(hop_node, {}).update(node_attr)

                        hop_edge_description = hop["edge"]
                        # 对多跳边补全属性
                        for graph in [graph1, graph2]:
                            for edge in graph:
                                if edge["relationship"] == hop_edge_description:
                                    edge_props = edge.get("edge_properties", {})
                                    entry.setdefault("multi_hop_edges_attrs", {}).setdefault(hop_edge_description, {}).update(edge_props)

                else:
                    print(f"Skipping entry (type: {type(entry)}) {entry} as it is not a dictionary or missing 'center' key.")

    return data






# 保存补全后的json文件到指定目录
def save_filled_json(filled_data, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 将补全后的数据保存到指定目录
    for i, data in enumerate(filled_data):
        output_file = os.path.join(output_directory, f"filled_data_{i + 1}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved filled JSON to: {output_file}")

# 主执行函数
def main():
    # 文件路径设置
    input_directory = "../dual_star_json"  # 输入目录路径
    graph1_path = "../graph1_enriched.json"  # graph1路径
    graph2_path = "../graph2_enriched.json"  # graph2路径
    output_directory = "../dual_star_json_with_attrs"  # 输出目录路径

    # 读取待补全数据
    data = load_json_from_directory(input_directory)

    # 读取graph1和graph2的数据
    graph1, graph2 = load_graph_data(graph1_path, graph2_path)

    # 补全节点和边的属性
    filled_data = fill_node_and_edge_attributes(data, graph1, graph2)

    # 保存补全后的数据
    save_filled_json(filled_data, output_directory)

if __name__ == "__main__":
    main()
