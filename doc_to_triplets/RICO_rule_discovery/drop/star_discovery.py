"""
第一类相关结构捕获
捕获图中可能的星型结构
将得到的所有星型结构放在同一个图中输出
"""

import json
from collections import defaultdict
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

def load_graphs(files):
    combined = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            combined.extend(data)
    return combined

def build_adjacency(triples):
    adjacency = defaultdict(list)
    for triple in triples:
        node1 = triple["node_1"]["name"]
        node2 = triple["node_2"]["name"]

        # 添加双向连接
        adjacency[node1].append((node2, triple))
        adjacency[node2].append((node1, triple))
    return adjacency

def extract_star_structures(triples):
    adjacency = build_adjacency(triples)
    stars = []

    for node, neighbors in adjacency.items():
        if len(neighbors) < 2:
            continue  # 至少两个连接才算星型中心

        is_star = True
        # 判断是否所有连接都只通过该中心节点（不能互相连接）
        neighbor_names = [n[0] for n in neighbors]
        for neighbor in neighbor_names:
            # 如果任何两个邻居之间也有边，就不是星型
            neighbor_edges = adjacency[neighbor]
            for n2, _ in neighbor_edges:
                if n2 != node and n2 in neighbor_names:
                    is_star = False
                    break
            if not is_star:
                break

        if is_star:
            star = {
                "center": node,
                "center_properties": {},
                "edges": [],
                "neighbors": []
            }
            for neighbor, edge in neighbors:
                star["edges"].append(edge)
                star["neighbors"].append(edge["node_1"] if edge["node_1"]["name"] != node else edge["node_2"])
                # 添加中心属性
                if edge["node_1"]["name"] == node:
                    star["center_properties"] = edge["node_1"]
                else:
                    star["center_properties"] = edge["node_2"]
            stars.append(star)
    return stars

def save_star_structures(stars, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stars, f, indent=2, ensure_ascii=False)

def convert_star_json_to_flat_triples(star_data):
    """
    将星型结构转换为扁平化三元组形式，供图可视化使用。
    """
    flat_data = []
    for star in star_data:
        for edge in star["edges"]:
            flat_data.append({
                "node_1": edge["node_1"],
                "node_2": edge["node_2"],
                "relationship": edge["relationship"]  # 注意这里是 relation 不是 relationship
            })
    return flat_data

def simplify_edge_labels_with_llm(relationships, llm):
    system_prompt = (
        "You are a helpful assistant. Your task is to simplify relationship phrases "
        "into short, clear, lowercase labels (e.g., 'works_with', 'married_to', 'located_in'). "
        "Avoid full sentences or long phrases. Use underscores to separate words."
    )

    joined = "\n".join(f"- {rel}" for rel in relationships)
    user_prompt = f"""Simplify the following relationship phrases:

{joined}

Return one simplified label per line in the same order."""

    response = llm.generate(user_message=user_prompt, system_message=system_prompt)
    simplified = [line.strip("- ").strip() for line in response.strip().splitlines() if line.strip()]
    return simplified

def visualize_property_graph_with_simplified_edges(json_data, llm):
    import networkx as nx
    import matplotlib.pyplot as plt
    import copy

    # 简化边的关系标签
    relationships = list({entry["relationship"] for entry in json_data})
    simplified_labels = simplify_edge_labels_with_llm(relationships, llm)
    rel_map = dict(zip(relationships, simplified_labels))

    # 打印关系映射（只有关系，没有属性）
    print("=== 三元组关系映射 ===")
    for orig, simp in rel_map.items():
        print(f"{orig}  -->  {simp}")
        print("*"*100)
    print("====================\n")

    # 构建图
    G = nx.DiGraph()
    label_colors = {
        "Person": "#87CEFA", "Company": "#FFD700", "Event": "#FFB6C1",
        "Department": "#98FB98", "Location": "#DDA0DD", "default": "#D3D3D3"
    }

    for entry in json_data:
        node1 = copy.deepcopy(entry["node_1"])
        node2 = copy.deepcopy(entry["node_2"])

        name1 = node1.pop("name", node1.pop("label", "Unknown"))
        name2 = node2.pop("name", node2.pop("label", "Unknown"))
        label1 = node1.get("label", "default")
        label2 = node2.get("label", "default")

        G.add_node(name1, label=label1)
        G.add_node(name2, label=label2)

        simplified_label = rel_map.get(entry["relationship"], "related")
        G.add_edge(name1, name2, label=simplified_label)

    # 可视化
    pos = nx.spring_layout(G, k=1.2)
    fig, ax = plt.subplots(figsize=(20, 14))

    node_colors = [label_colors.get(G.nodes[n].get("label", "default"), "#D3D3D3") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, edgecolors="black", ax=ax)

    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=50, edge_color='gray', ax=ax)

    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', ax=ax)

    plt.title("Knowledge Graph with Simplified Edge Labels", fontsize=18)
    plt.axis("off")
    plt.savefig("star_graph/merged_graph_simple.png", dpi=300)
    plt.show()

def visualize_star_structures(file_path, llm):
    with open(file_path, "r", encoding="utf-8") as f:
        star_data = json.load(f)

    flat_triples = convert_star_json_to_flat_triples(star_data)
    visualize_property_graph_with_simplified_edges(flat_triples, llm)

def main():
    input_files = ["graph1_enriched.json", "graph2_enriched.json"]
    output_file = "../star_graph/star_structures.json"

    triples = load_graphs(input_files)
    stars = extract_star_structures(triples)
    save_star_structures(stars, output_file)

    print(f"[DONE] 提取到 {len(stars)} 个星型结构，已写入 {output_file}")

    visualize_star_structures("../star_graph/star_structures.json", llm)

if __name__ == "__main__":
    main()
