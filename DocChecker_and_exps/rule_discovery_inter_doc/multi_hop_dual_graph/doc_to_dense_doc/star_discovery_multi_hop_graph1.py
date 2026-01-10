"""
首先执行，生成graph1的星型结构，json保存在star_json_graph1文件夹下，graph保存在star_graph1文件夹下
生成多跳星型图，可以借助merge_star_structures_with_hops(stars, max_hops=5)这一函数设置最大跳数
分开保留各个得到的星型图，并进行可视化
输出文件目录为star_graph1和star_json_graph1
"""
import json
from collections import defaultdict
import os
import networkx as nx
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

def load_graph(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 调试输出，检查每个条目的内容
    for i, entry in enumerate(data):
        if "relationship" not in entry:
            print(f"[WARNING] Entry {i} is missing 'relationship': {entry}")

    return data

def is_overlapping(triple1, triple2):
    """ 判断两个三元组的 text_span 是否重叠 """
    span1 = triple1["edge_properties"]["text_span"]
    span2 = triple2["edge_properties"]["text_span"]

    # 判断两个文本跨度是否重叠
    return not (span1["end"] < span2["start"] or span2["end"] < span1["start"])


def build_star_structure(center_node, triples):
    """ 为给定的实体和其相关三元组创建星型结构 """
    neighbors = [triple["node_2"]["name"] if triple["node_1"]["name"] == center_node else triple["node_1"]["name"] for triple in triples]
    edges = [triple["relationship"] for triple in triples]

    return {
        "center": center_node,
        "neighbors": neighbors,
        "edges": edges
    }

def combine_triples_with_same_entity(triples, max_hops=1):
    """ 合并包含相同实体的三元组，不再判断上下文（相邻性） """
    entity_triples_map = defaultdict(list)
    for triple in triples:
        node1_name = triple["node_1"]["name"]
        node2_name = triple["node_2"]["name"]

        # 根据节点名称将三元组分类
        entity_triples_map[node1_name].append(triple)
        entity_triples_map[node2_name].append(triple)

    merged_stars = []

    # 遍历每个实体的三元组
    for entity, entity_triples in entity_triples_map.items():
        # 为每个实体创建一个星型结构
        star = build_star_structure(entity, entity_triples)
        merged_stars.append(star)

    return merged_stars


def get_neighbors_with_hops(node, hop_count, max_hops, stars, visited_nodes=set(), neighbors_set=set(), multi_hop_neighbors=set()):
    """ 获取指定节点的所有邻居，最多最大跳数 """
    if hop_count > max_hops or node in visited_nodes:
        return visited_nodes, multi_hop_neighbors

    visited_nodes.add(node)

    # 遍历所有的星型结构
    for star in stars:
        if star["center"] == node:
            for neighbor, edge in zip(star["neighbors"], star["edges"]):
                # 如果这是第一跳邻居，直接添加到 neighbors_set 中
                if hop_count == 1:
                    neighbors_set.add(neighbor)
                    # 添加该边的关系到 multi_hop_neighbors 中
                    multi_hop_neighbors.add((neighbor, hop_count, edge))
                # 如果是多跳邻居，添加到 multi_hop_neighbors
                elif hop_count > 1:
                    multi_hop_neighbors.add((neighbor, hop_count, edge))
                # 递归获取邻居的邻居
                visited_nodes, multi_hop_neighbors = get_neighbors_with_hops(neighbor, hop_count + 1, max_hops, stars, visited_nodes, neighbors_set, multi_hop_neighbors)

    return visited_nodes, multi_hop_neighbors


def remove_single_hop_neighbors(star):
    """ 删除多跳邻居中的跳数为1的邻居，并删除与 neighbors 中节点重复的 multi_hop_neighbors 节点 """
    # 删除 hop_count 为 1 的邻居
    star["multi_hop_neighbors"] = [
        neighbor_info for neighbor_info in star["multi_hop_neighbors"] if neighbor_info["hop_count"] > 1
    ]

    # 获取 neighbors 中的所有节点
    neighbor_nodes = set(star["neighbors"])

    # 遍历 multi_hop_neighbors 删除重复的节点
    star["multi_hop_neighbors"] = [
        neighbor_info for neighbor_info in star["multi_hop_neighbors"]
        if neighbor_info["neighbor"] not in neighbor_nodes
    ]

    return star


def remove_duplicate_center(data):
    """
    遍历数据，删除重复的中心节点，并将重复节点的信息合并到第一个出现的该中心节点中。
    """
    centers_to_remove = {}  # 用于记录每个中心节点的索引列表

    # 遍历数据找到所有重复的中心节点
    for i, entry in enumerate(data):
        center = entry['center']
        if center not in centers_to_remove:
            centers_to_remove[center] = []
        centers_to_remove[center].append(i)

    # 删除重复中心节点，并合并信息
    for center, indices in centers_to_remove.items():
        if len(indices) > 1:
            # 如果该中心节点出现多次
            main_index = indices[0]  # 选择第一次出现的中心节点
            # 遍历所有该中心节点的重复项，合并信息
            for idx in indices[1:]:
                # 合并邻居
                for neighbor in data[idx]['neighbors']:
                    if neighbor not in data[main_index]['neighbors']:
                        data[main_index]['neighbors'].append(neighbor)
                # 合并边缘
                for edge in data[idx]['edges']:
                    if edge not in data[main_index]['edges']:
                        data[main_index]['edges'].append(edge)
                # 合并多跳邻居
                for multi_hop in data[idx]['multi_hop_neighbors']:
                    if multi_hop not in data[main_index]['multi_hop_neighbors']:
                        data[main_index]['multi_hop_neighbors'].append(multi_hop)
            # 删除重复项
            for idx in reversed(indices[1:]):  # 从后往前删除，避免索引变化
                data.pop(idx)

    return data



def merge_star_structures_with_hops(stars, max_hops=5):
    """ 合并星型结构，并根据最大跳数添加邻居 """
    merged_stars = []
    entity_to_star_map = {}
    all_nodes_in_merged = set()
    node_neighbors_with_hops = defaultdict(list)

    # 遍历每个星型结构，进行合并
    for star in stars:
        center = star["center"]
        neighbors = set(star["neighbors"])
        edges = star["edges"]

        # 获取当前星型结构的多跳邻居，返回的应当是多跳邻居和更新的visited_nodes
        visited_nodes, multi_hop_neighbors = get_neighbors_with_hops(center, 1, max_hops, stars, set(), set(), set())

        # 保证不重复添加中心节点到 multi_hop_neighbors
        multi_hop_neighbors = [item for item in multi_hop_neighbors if item[0] != center]

        # 将新的邻居和边添加到当前的星型结构
        for neighbor, edge in zip(neighbors, edges):
            if neighbor not in all_nodes_in_merged:
                star["neighbors"].append(neighbor)
                star["edges"].append(edge)
                all_nodes_in_merged.add(neighbor)

        # 将多跳邻居添加到星型结构中
        star["multi_hop_neighbors"] = [{
            "neighbor": neighbor,
            "hop_count": hop_count,
            "edge": edge
        } for neighbor, hop_count, edge in multi_hop_neighbors]

        # 删除 "multi_hop_neighbors" 中相同 neighbor 的重复项
        seen_neighbors = set()
        unique_multi_hop_neighbors = []
        for mh_neighbor in star["multi_hop_neighbors"]:
            if mh_neighbor["neighbor"] not in seen_neighbors:
                seen_neighbors.add(mh_neighbor["neighbor"])
                unique_multi_hop_neighbors.append(mh_neighbor)

        # 更新去重后的 multi_hop_neighbors
        star["multi_hop_neighbors"] = unique_multi_hop_neighbors

        # 删除单跳邻居
        star = remove_single_hop_neighbors(star)

        # 现在，检查这个新星型结构的中心节点是否是任何已有星型结构的邻居或多跳邻居
        # 如果是，则需要将它合并到已经存在的星型结构中
        if center in entity_to_star_map:
            existing_star = entity_to_star_map[center]

            # 合并新星型结构与已有星型结构
            existing_neighbors = set(existing_star["neighbors"])
            existing_edges = existing_star["edges"]

            # 合并邻居和边
            existing_neighbors.update(neighbors)
            existing_edges.extend([edge for edge in edges if edge not in existing_edges])

            # 更新已有星型结构
            existing_star["neighbors"] = list(existing_neighbors)
            existing_star["edges"] = existing_edges

            # 合并多跳邻居
            existing_star["multi_hop_neighbors"].extend(star["multi_hop_neighbors"])

            # 删除当前已经被合并的星型结构
            continue

        # 将新合并的星型结构加入到结果列表中
        merged_stars.append(star)
        entity_to_star_map[center] = star
        all_nodes_in_merged.add(center)

    merged_stars = remove_duplicate_center(merged_stars)

    return merged_stars



def split_star_structures(data):
    # 用一个字典来保存每个中心节点对应的星型结构
    centers = {}

    for entry in data:
        center_name = entry['center']

        if center_name not in centers:
            centers[center_name] = []

        # 将当前中心的星型结构添加到对应的列表中
        centers[center_name].append(entry)

    # 创建保存文件的目录（如果不存在）
    output_dir = "star_json_graph1"
    os.makedirs(output_dir, exist_ok=True)

    # 将每个中心节点的星型结构保存为独立的JSON文件
    for center_name, structures in centers.items():
        # 生成JSON文件名，并保存到star_json_graph1目录下
        file_name = os.path.join(output_dir, f"{center_name.replace(' ', '_')}_star_structure.json")

        # 将数据写入JSON文件
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(structures, f, ensure_ascii=False, indent=4)

        print(f"File '{file_name}' has been created.")


def simplify_edge_labels_with_llm(relationships, llm):
    """ 使用 LLM 简化关系边标签 """
    system_prompt = (
        "You are a helpful assistant. Your task is to simplify relationship phrases "
        "into short, clear, lowercase labels (e.g., 'works_with', 'married_to', 'located_in'). "
        "Avoid full sentences or long phrases. Use underscores to separate words."
    )

    # 将关系连接成一个字符串
    joined = "\n".join(f"- {rel}" for rel in relationships)
    user_prompt = f"""Simplify the following relationship phrases:

{joined}

Return one simplified label per line in the same order."""

    # 获取 LLM 的响应
    response = llm.generate(user_message=user_prompt, system_message=system_prompt)

    # 处理响应并返回简化后的标签
    simplified = [line.strip("- ").strip() for line in response.strip().splitlines() if line.strip()]
    return simplified

def visualize_star_structure_with_hops(json_data, output_file, title="Star Structure", llm=None):
    """ 可视化包含多跳邻居的星型结构，并限制下一跳节点只能作为出边，同时将关系放置在侧边栏 """
    # 检查 json_data 是否是字典类型，并提取中心节点、邻居节点、边和多跳邻居
    if isinstance(json_data, dict):
        center = json_data.get("center")
        neighbors = json_data.get("neighbors", [])
        edges = json_data.get("edges", [])
        multi_hop_neighbors = json_data.get("multi_hop_neighbors", [])
    else:
        print("Error: json_data should be a dictionary with 'center', 'neighbors', and 'multi_hop_neighbors' keys.")
        return

    if not center or not neighbors or not edges:
        print("Error: Missing required data in json_data. Ensure 'center', 'neighbors', and 'edges' are present.")
        return

    # 使用 LLM 简化边的标签（如果提供了 LLM）
    if llm:
        simplified_edge_labels = simplify_edge_labels_with_llm(edges, llm)
    else:
        simplified_edge_labels = edges  # 如果没有 LLM，则使用原始边标签

    # 创建一个有向图
    G = nx.DiGraph()

    # 添加中心节点和邻居节点
    G.add_node(center, label="center")
    for neighbor in neighbors:
        G.add_node(neighbor, label="neighbor")

    # 添加边，假设每条边的关系为边标签
    for simplified_edge, neighbor in zip(simplified_edge_labels, neighbors):
        G.add_edge(center, neighbor, label=simplified_edge)

    # 记录多跳邻居的前一跳邻居
    prev_neighbors = neighbors  # 初始化为直接邻居节点
    all_hop_neighbors = {0: neighbors}  # 用于存储各层级的邻居

    # 添加多跳邻居并与前一跳邻居相连，确保下一跳节点仅为出边
    for multi_hop in multi_hop_neighbors:
        neighbor = multi_hop["neighbor"]
        hop_count = multi_hop["hop_count"]
        edge = multi_hop["edge"]

        # 推导出上一跳邻居（基于当前 hop_count 和前一跳的邻居）
        if hop_count > 1 and hop_count - 1 in all_hop_neighbors:
            previous_hop_neighbor = all_hop_neighbors[hop_count - 1]
        else:
            previous_hop_neighbor = neighbors  # 如果是第一跳，则直接连到中心节点

        # 新增多跳邻居节点
        G.add_node(neighbor, label=f"hop_{hop_count}")

        # 确保多跳邻居与前一跳邻居相连，但不允许反向连接
        if hop_count - 1 in all_hop_neighbors:
            prev_neighbors = all_hop_neighbors[hop_count - 1]
            for prev_neighbor in prev_neighbors:
                # 只添加出边，不添加入边
                if not G.has_edge(prev_neighbor, neighbor):  # 确保没有重复的边
                    G.add_edge(prev_neighbor, neighbor, label=edge)

        # 连接上一跳邻居到当前多跳邻居
        if previous_hop_neighbor != center:  # 确保上一跳不是中心节点
            for prev_hop in previous_hop_neighbor:
                if not G.has_edge(prev_hop, neighbor):
                    G.add_edge(prev_hop, neighbor, label=edge)

        # 更新各层级邻居
        if hop_count not in all_hop_neighbors:
            all_hop_neighbors[hop_count] = []
        all_hop_neighbors[hop_count].append(neighbor)

    # 设置图形布局
    pos = nx.spring_layout(G, k=1.2)
    fig, ax = plt.subplots(figsize=(20, 14))

    # 节点颜色
    label_colors = {
        "center": "#FFD700",  # 黄金色
        "neighbor": "#87CEFA",  # 淡蓝色
        "multi_hop_neighbor": "#32CD32",  # 多跳邻居，绿色
        "default": "#D3D3D3"  # 默认灰色
    }

    # 节点颜色映射
    node_colors = [
        label_colors.get(G.nodes[n].get("label", "default"), "#D3D3D3") for n in G.nodes()
    ]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, edgecolors="black", ax=ax)

    # 绘制节点标签
    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)

    # 绘制边
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=50, edge_color='gray', ax=ax)

    # 绘制边标签（我们将这些关系移到侧边栏）
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}

    # 计算侧边栏位置
    sidebar_x = 1.05  # 侧边栏x位置
    sidebar_y_start = 0.95  # 侧边栏y起始位置
    label_margin = 0.02  # 标签的垂直间距
    max_label_height = 0.85  # 侧边栏最大高度

    # 添加关系标签到侧边栏
    for (u, v), label in edge_labels.items():
        if label:  # 如果有标签，绘制在侧边栏
            ax.text(
                sidebar_x, sidebar_y_start, label, fontsize=8, ha='left', va='top',
                transform=ax.transAxes, color="darkred"
            )
            sidebar_y_start -= label_margin  # 每个标签下移

    # 设置标题
    plt.title(title, fontsize=18)
    plt.axis("off")

    # 确保创建输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)  # 确保目标目录存在

    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] 图已保存为: {output_file}")






def visualize_all_star_structures(input_dir="star_json_graph1", llm_model=None):
    """ 遍历目录下的所有星型结构文件并可视化 """
    # 获取 star_json_graph1 目录下的所有 JSON 文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)

            # 读取 JSON 文件
            with open(file_path, "r", encoding="utf-8") as f:
                star_data = json.load(f)

            # 如果 star_data 是一个列表（多个星型结构），则逐个处理
            if isinstance(star_data, list):
                for star in star_data:
                    # 获取中心节点名称
                    center_name = star.get("center", "Unknown Center")
                    title = f"Star Structure for {center_name}"
                    output_image = f"./star_graph1/{filename.replace('.json', '_{center_name}.png')}"

                    visualize_star_structure_with_hops(star, output_image, title, llm_model)
                    print(f"可视化图像已保存: {output_image}")
            else:
                # 如果是单个星型结构字典，直接处理
                center_name = star_data.get("center", "Unknown Center")
                title = f"Star Structure for {center_name}"
                output_image = f"./star_graph1/{filename.replace('.json', '.png')}"

                visualize_star_structure_with_hops(star_data, output_image, title, llm_model)
                print(f"可视化图像已保存: {output_image}")



def main():
    # 输入文件
    input_file = "graph1_enriched.json"
    print(f"[PROCESSING] 提取星型结构：{input_file}")

    # 加载三元组数据
    triples = load_graph(input_file)
    print(f"[DEBUG] Loaded JSON data type: {type(triples)}")
    print(f"[DEBUG] Loaded JSON data content: {triples[:5]}")  # 打印前5条数据检查格式

    # 合并三元组
    stars = combine_triples_with_same_entity(triples)

    # 合并星型结构
    merged_stars = merge_star_structures_with_hops(stars)

    # 调用split_star_structures保存独立的星型结构文件
    split_star_structures(merged_stars)

    # 可视化所有从 star_json_graph1 目录中读取的星型结构
    visualize_all_star_structures(input_dir="star_json_graph1")


if __name__ == "__main__":
    main()
