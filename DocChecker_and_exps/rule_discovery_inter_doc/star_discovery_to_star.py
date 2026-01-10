"""
在star_discovery_to_dual_star.py之前执行
第一种相关关系捕获，分开提取G1和G2中的星型结构
将得到的所有星型结构分开呈现
会在./star_json和./star_graph目录下分别创建graph1和graph2两个目录，存储分别从G1和G2中找到的星型结构
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

"""
提取中心到节点多跳的星型结构
"""

def extract_star_structures(triples, max_hops=1):
    adjacency = build_adjacency(triples)
    stars = []

    for node in adjacency.keys():
        # 用多跳函数获取邻居
        neighbors = get_neighbors_with_hops(adjacency, node, max_hops)

        if len(neighbors) < 1:
            continue  # 至少1个邻居

        # 判断邻居间是否有连接
        neighbor_names = [n for n in neighbors]
        is_star = True
        for neighbor in neighbor_names:
            neighbor_edges = adjacency.get(neighbor, [])
            for n2, _ in neighbor_edges:
                if n2 != node and n2 in neighbor_names:
                    is_star = False
                    break
            if not is_star:
                break

        if is_star:
            star = {
                "center": node,
                "center_properties": get_node_properties(node, triples),
                "edges": [],
                "neighbors": []
            }
            for neighbor in neighbor_names:
                # 选出连接中心和邻居的边
                edges = [edge for n, edge in adjacency[node] if n == neighbor]
                star["edges"].extend(edges)
                # 找邻居的节点属性（根据 triples 或构建的节点字典）
                star["neighbors"].append(get_node_properties(neighbor, triples))

            stars.append(star)

    return stars

def get_neighbors_with_hops(adjacency, center_node, max_hops):
    visited = set([center_node])
    frontier = set([center_node])
    neighbors = set()

    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            for n, _ in adjacency.get(node, []):
                if n not in visited:
                    neighbors.add(n)
                    next_frontier.add(n)
                    visited.add(n)
        frontier = next_frontier

    return list(neighbors)

def get_node_properties(node_name, triples):
    # 从 triples 找到这个节点的属性
    for triple in triples:
        if triple["node_1"]["name"] == node_name:
            return triple["node_1"]
        if triple["node_2"]["name"] == node_name:
            return triple["node_2"]
    return {"name": node_name}


def save_star_structures(stars, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stars, f, indent=2, ensure_ascii=False)

def convert_star_json_to_flat_triples(star_data):
    flat_data = []

    for star in star_data:
        center_node = star.get("center")
        if not center_node:
            continue

        center_name = center_node

        for edge in star.get("edges", []):
            n1 = edge.get("node_1")
            n2 = edge.get("node_2")

            if n1.get("name") == center_name:
                source = n1
                target = n2
            elif n2.get("name") == center_name:
                source = n2
                target = n1
            else:
                source = n1
                target = n2

            # ✅ 保留边的所有属性（除了 node_1 和 node_2）
            edge_attributes = {k: v for k, v in edge.items() if k not in ("node_1", "node_2")}

            flat_data.append({
                "node_1": source,
                "node_2": target,
                **edge_attributes  # 展开边的属性
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

def visualize_property_graph_with_simplified_edges(json_data, llm, title="Knowledge Graph", output_file="graph.png"):
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

    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[SAVED] 图已保存为: {output_file}")


def visualize_star_structures(star_data, llm, output_prefix="graph1"):
    print(f"[INFO] 正在可视化 {output_prefix} 的 {len(star_data)} 个星型结构")

    output_graph_dir = f"./star_graph/{output_prefix}"
    output_json_dir = f"./star_json/{output_prefix}"
    os.makedirs(output_graph_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    for idx, star in enumerate(star_data, 1):
        center = star["center"]
        flat_triples = convert_star_json_to_flat_triples([star])  # 单个星结构

        title = f"Star Structure around '{center}'"
        safe_center = center.replace(" ", "_").replace("/", "_")
        output_image = f"{output_graph_dir}/star_graph_{idx}_{safe_center}.png"
        output_json = f"{output_json_dir}/star_graph_{idx}_{safe_center}.json"

        # ✅ 保存 JSON 数据
        with open(output_json, "w", encoding="utf-8") as jf:
            json.dump(flat_triples, jf, indent=2, ensure_ascii=False)

        # ✅ 保存图像
        visualize_property_graph_with_simplified_edges(
            flat_triples,
            llm,
            title=title,
            output_file=output_image
        )


def main():
    # 输入文件分开处理
    input_files = {
        "graph1": "graph1_enriched.json",
        "graph2": "graph2_enriched.json"
    }

    # 分别处理两个图
    for graph_name, file_path in input_files.items():
        print(f"\n[PROCESSING] 提取 {graph_name} 的星型结构：{file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            triples = json.load(f)

        stars = extract_star_structures(triples)
        output_star_json = f"star_graph/star_structures_{graph_name}.json"

        save_star_structures(stars, output_star_json)
        print(f"[SAVED] {graph_name} 提取到 {len(stars)} 个星型结构，已写入 {output_star_json}")

        visualize_star_structures(stars, llm, output_prefix=graph_name)

if __name__ == "__main__":
    main()
