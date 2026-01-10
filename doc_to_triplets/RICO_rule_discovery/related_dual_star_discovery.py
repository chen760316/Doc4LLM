"""
第三种相关关系捕获，提取隶属于同一三元组的主语和宾语作为双星结构的两个星型的中心
捕获图中可能的双星结构，其中两个中心构成三元组的连接关系，双星结构的两个星型属于有关联的星型结构
呈现总的双星结构，分开呈现所有的双星结构
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
from collections import defaultdict


# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

def load_graphs(file_list):
    all_triples = []
    for fpath in file_list:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_triples.extend(data)
    return all_triples

def build_adjacency(triples):
    adj = defaultdict(list)
    for triple in triples:
        n1 = triple["node_1"]["name"]
        n2 = triple["node_2"]["name"]
        adj[n1].append((n2, triple))
        adj[n2].append((n1, triple))
    return adj

def get_node_properties(node_name, triples):
    for triple in triples:
        if triple["node_1"]["name"] == node_name:
            return triple["node_1"]
        if triple["node_2"]["name"] == node_name:
            return triple["node_2"]
    return {"name": node_name}

def extract_all_entities(triples):
    entities = defaultdict(list)
    for triple in triples:
        entities[triple["node_1"]["name"]].append(triple)
        entities[triple["node_2"]["name"]].append(triple)
    return entities

def build_star_structure(center, adjacency, triples):
    """
    给定中心节点，构建星型结构：中心 + 所有直接邻居和对应边
    """
    star = {
        "center": center,
        "center_properties": get_node_properties(center, triples),
        "neighbors": [],
        "edges": []
    }
    neighbors = adjacency.get(center, [])
    for neighbor_name, edge in neighbors:
        star["neighbors"].append(get_node_properties(neighbor_name, triples))
        star["edges"].append(edge)
    return star

def convert_star_to_flat_triples(star, unique_prefix):
    flat_triples = []
    center_uid = f"{star['center']}__{unique_prefix}"

    for edge in star["edges"]:
        n1 = copy.deepcopy(edge["node_1"])
        n2 = copy.deepcopy(edge["node_2"])

        if n1["name"] == star["center"]:
            n1["name"] = center_uid
        if n2["name"] == star["center"]:
            n2["name"] = center_uid

        # ✅ 新增：保留所有原始边的额外属性（除了 node_1, node_2, relationship）
        extra_attrs = {
            k: v for k, v in edge.items()
            if k not in {"node_1", "node_2", "relationship"}
        }

        flat_triples.append({
            "node_1": n1,
            "node_2": n2,
            "relationship": edge.get("relationship", "related"),
            **extra_attrs  # ✅ 添加边的附加属性
        })
    return flat_triples


def generate_virtual_links(star_uids):
    """
    生成虚拟边，连接所有星型结构的中心节点，表示同一实体
    """
    virtual_edges = []
    for i in range(len(star_uids)):
        for j in range(i + 1, len(star_uids)):
            virtual_edges.append({
                "node_1": {"name": star_uids[i], "label": "VirtualCenter"},
                "node_2": {"name": star_uids[j], "label": "VirtualCenter"},
                "relationship": "related"
            })
    return virtual_edges


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
        G.add_edge(name1, name2, label=simplified_label, orig_rel=entry["relationship"])

    # 生成布局，保证 pos 不为 None，参考之前的建议
    pos = None
    if "Dual-Star Structure:" in title:
        centers = [n for n, d in G.nodes(data=True) if d.get("label") == "VirtualCenter"]
        if len(centers) >= 2:
            center1, center2 = centers[0], centers[1]

            group1 = set([center1])
            group2 = set([center2])
            for nbr in G.neighbors(center1):
                group1.add(nbr)
            for nbr in G.neighbors(center2):
                group2.add(nbr)

            pos = {}
            for i, node in enumerate(group1):
                pos[node] = (0, -i)
            for i, node in enumerate(group2):
                pos[node] = (3, -i)

            others = set(G.nodes()) - group1 - group2
            if others:
                mid_pos = nx.spring_layout(G.subgraph(others), seed=42)
                for node, p in mid_pos.items():
                    pos[node] = (1.5, p[1])
        else:
            pos = nx.spring_layout(G, k=1.2, seed=42)
    else:
        pos = nx.spring_layout(G, k=1.2, seed=42)

    fig, ax = plt.subplots(figsize=(20, 14))

    # 节点颜色
    node_colors = [label_colors.get(G.nodes[n].get("label", "default"), "#D3D3D3") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, edgecolors="black", ax=ax)

    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)

    # 先画非 related 边，单向箭头
    non_related_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("orig_rel") != "related"]
    nx.draw_networkx_edges(G, pos, edgelist=non_related_edges, arrowstyle='-|>', arrowsize=40, edge_color='gray', ax=ax)

    # 再画 related 边，双向箭头
    related_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("orig_rel") == "related"]
    nx.draw_networkx_edges(G, pos, edgelist=related_edges, arrowstyle='<->', arrowsize=45, edge_color='red', width=2, ax=ax)

    # 边标签
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', ax=ax)

    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[SAVED] 图已保存为: {output_file}")



def visualize_graph(flat_triples, llm, title, output_file):
    visualize_property_graph_with_simplified_edges(flat_triples, llm, title=title, output_file=output_file)

def build_star_structure_outgoing(center, adjacency, triples):
    star = {
        "center": center,
        "center_properties": get_node_properties(center, triples),
        "neighbors": [],
        "edges": []
    }
    neighbors = adjacency.get(center, [])
    for neighbor_name, edge in neighbors:
        star["neighbors"].append(get_node_properties(neighbor_name, triples))
        star["edges"].append(edge)
    return star

def build_adjacency_outgoing(triples):
    adj = defaultdict(list)
    for triple in triples:
        n1 = triple["node_1"]["name"]
        n2 = triple["node_2"]["name"]
        adj[n1].append((n2, triple))  # 只添加中心节点指向邻居的边
    return adj


def main():
    input_files = ["graph1_enriched.json", "graph2_enriched.json"]
    output_file = "./related_dual_star_graph/related_dual_star_graph.json"
    graph_output_dir = "./related_dual_star_graph"
    json_output_dir = "./related_dual_star_json"
    os.makedirs(graph_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    triples = load_graphs(input_files)
    adjacency = build_adjacency_outgoing(triples)

    # 找出所有实体出现次数 >= 2 的实体名（即相同实体）
    entities = extract_all_entities(triples)
    repeated_entities = {ename: entities[ename] for ename in entities if len(entities[ename]) >= 2}

    print(f"[INFO] 发现 {len(repeated_entities)} 个重复实体")

    # 为每个重复实体构建对应的星型结构列表
    entity_stars = {}
    for entity_name in repeated_entities:
        # 可能有多个星型结构? 这里每个中心只有一个星型结构（因为中心唯一）
        star = build_star_structure(entity_name, adjacency, triples)
        entity_stars[entity_name] = star

    # 生成双星结构：两两组合重复实体
    all_flat_triples = []
    virtual_links_all = []

    repeated_names = list(entity_stars.keys())

    for i in range(len(repeated_names)):
        for j in range(i + 1, len(repeated_names)):
            c1 = repeated_names[i]
            c2 = repeated_names[j]

            star1 = entity_stars[c1]
            star2 = entity_stars[c2]

            uid1 = f"{c1}__S{i}"
            uid2 = f"{c2}__S{j}"

            # 替换非法字符（如空格、斜杠）
            safe_uid1 = uid1.replace(" ", "_").replace("/", "_")
            safe_uid2 = uid2.replace(" ", "_").replace("/", "_")

            flat_star1 = convert_star_to_flat_triples(star1, unique_prefix=f"S{i}")
            flat_star2 = convert_star_to_flat_triples(star2, unique_prefix=f"S{j}")

            combined = flat_star1 + flat_star2

            # 添加虚拟边，连接两个中心
            virtual_edge = {
                "node_1": {"name": uid1, "label": "VirtualCenter"},
                "node_2": {"name": uid2, "label": "VirtualCenter"},
                "relationship": "related"
            }
            combined.append(virtual_edge)

            all_flat_triples.extend(combined)
            virtual_links_all.append(virtual_edge)

            # === 可视化 PNG 图 ===
            title = f"Dual-Star Structure: {c1} ↔ {c2}"
            png_path = f"{graph_output_dir}/dual_star_{safe_uid1}_vs_{safe_uid2}.png"
            visualize_graph(combined, llm, title, png_path)

            # ✅ 保存为 JSON 文件 ===
            json_path = f"{json_output_dir}/dual_star_{safe_uid1}_vs_{safe_uid2}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(combined, jf, indent=2, ensure_ascii=False)

    # 生成所有虚拟链接的总图
    all_uids = [f"{name}__S{idx}" for idx, name in enumerate(repeated_names)]
    virtual_links_all = generate_virtual_links(all_uids)
    all_flat_triples.extend(virtual_links_all)

    visualize_graph(all_flat_triples, llm,
                    title="All Dual-Star Structures with Virtual Links",
                    output_file="./related_dual_star_graph/dual_star_virtual_links.png")

    # 可选：保存所有星型结构数据
    stars_to_save = list(entity_stars.values())
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stars_to_save, f, indent=2, ensure_ascii=False)
    print(f"[DONE] 星型结构已保存到 {output_file}")

if __name__ == "__main__":
    main()
