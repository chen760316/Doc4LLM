"""
原始的进行星型图提取的代码
暂时useless
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
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
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


def save_star_structures(stars, output_file):
    # 确保目录存在，如果不存在则创建
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # 保存星型结构数据到指定文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stars, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] 星型结构已保存到: {output_file}")

def visualize_star_structure(json_data, llm, title="Star Structure", output_file="star_graph.png"):
    import networkx as nx
    import matplotlib.pyplot as plt

    # 调试输出，查看 json_data 的实际内容
    print(f"[DEBUG] json_data: {json_data}")

    # 检查 json_data 是否是字典类型，并提取中心节点、邻居节点和边
    if isinstance(json_data, dict):
        center = json_data.get("center")
        neighbors = json_data.get("neighbors", [])
        edges = json_data.get("edges", [])
    else:
        print("Error: json_data should be a dictionary with 'center', 'neighbors', and 'edges' keys.")
        return

    if not center or not neighbors or not edges:
        print("Error: Missing required data in json_data. Ensure 'center', 'neighbors', and 'edges' are present.")
        return

    # 创建一个有向图
    G = nx.DiGraph()

    # 添加中心节点和邻居节点
    G.add_node(center, label="center")
    for neighbor in neighbors:
        G.add_node(neighbor, label="neighbor")

    # 添加边，假设每条边的关系为边标签
    for edge, neighbor in zip(edges, neighbors):
        G.add_edge(center, neighbor, label=edge)

    # 设置图形布局
    pos = nx.spring_layout(G, k=1.2)
    fig, ax = plt.subplots(figsize=(20, 14))

    # 节点颜色
    label_colors = {
        "center": "#FFD700",  # 黄金色
        "neighbor": "#87CEFA",  # 淡蓝色
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

    # 绘制边标签
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', ax=ax)

    # 设置标题
    plt.title(title, fontsize=18)
    plt.axis("off")

    # 保存图像
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[SAVED] 图已保存为: {output_file}")


def visualize_star_structures(star_data, llm, output_prefix="merged_graph"):
    print(f"[INFO] 正在可视化星型结构：{len(star_data)} 个")

    # 创建输出目录
    output_graph_dir = f"./star_graph/{output_prefix}"
    output_json_dir = f"./star_json/merged_json"
    os.makedirs(output_graph_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    # 为每个星型结构生成图和JSON
    for idx, star in enumerate(star_data, 1):
        center = star["center"]
        title = f"Star Structure around '{center}'"
        output_image = f"{output_graph_dir}/star_graph_{idx}_{center}.png"
        output_json = f"{output_json_dir}/star_graph_{idx}_{center}.json"

        # 保存星型结构的JSON文件
        with open(output_json, "w", encoding="utf-8") as jf:
            json.dump(star, jf, indent=2, ensure_ascii=False)
        print(f"[SAVED] 星型结构的 JSON 已保存到: {output_json}")

        # 可视化并保存星型结构图
        visualize_star_structure(
            star,
            llm,
            title=title,
            output_file=output_image
        )

def merge_star_structures(stars):
    """ 合并相邻的星型结构，如果中心节点出现在其他星型结构的边上 """
    # 创建一个字典，存储已合并的星型结构
    merged_stars = []

    # 用于存储每个实体的合并情况
    entity_to_star_map = {}

    for star in stars:
        center = star["center"]
        neighbors = set(star["neighbors"])
        edges = star["edges"]

        # 如果中心节点已经在其他星型结构中，进行合并
        if center in entity_to_star_map:
            existing_star = entity_to_star_map[center]

            # 合并邻居节点和边（避免重复）
            for neighbor, edge in zip(neighbors, edges):
                if neighbor not in existing_star["neighbors"]:
                    existing_star["neighbors"].append(neighbor)
                    existing_star["edges"].append(edge)
        else:
            # 如果中心节点不在其他星型结构中，直接添加
            merged_stars.append(star)
            entity_to_star_map[center] = star

    return merged_stars

def main():
    # 输入文件
    input_file = "../merged_enriched_graph.json"
    print(f"[PROCESSING] 提取星型结构：{input_file}")

    # 加载三元组数据
    triples = load_graph(input_file)
    print(f"[DEBUG] Loaded JSON data type: {type(triples)}")
    print(f"[DEBUG] Loaded JSON data content: {triples[:5]}")  # 打印前5条数据检查格式

    # 合并三元组
    stars = combine_triples_with_same_entity(triples)

    # 合并星型结构
    merged_stars = merge_star_structures(stars)

    # 更新输出路径
    output_star_json = "star_json/merged_graph/star_structures.json"

    # 保存星型结构
    save_star_structures(merged_stars, output_star_json)

    # 可视化星型结构
    visualize_star_structures(merged_stars, llm, output_prefix="merged_graph")


if __name__ == "__main__":
    main()
