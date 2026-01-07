"""
第二种相关关系捕获
捕获图中可能的双星结构
呈现总的双星结构，分开呈现所有的双星结构
只要识别出了相同实体，就会将两个相同实体作为双星结构的两个中心，别的从中心结构延伸出去的节点保留在双星结构中
MAX_DUAL_STARS_PER_ENTITY = 5：限制每个中心最多生成的双星数量为5
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
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

def load_graphs(file_list):
    all_triples = []
    for fpath in file_list:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_triples.extend(data)  # ✅ 展平添加每个文件的三元组
            else:
                raise ValueError(f"Expected a list of triples in {fpath}, but got {type(data)}")
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

def extract_all_entities(triples, file_ids):
    entity_to_files = defaultdict(set)
    for triple, fid in zip(triples, file_ids):
        entity_to_files[triple["node_1"]["name"]].add(fid)
        entity_to_files[triple["node_2"]["name"]].add(fid)
    return entity_to_files


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

        # ✅ 修改名称为唯一 ID（用于虚拟中心）
        if n1["name"] == star["center"]:
            n1["name"] = center_uid
        if n2["name"] == star["center"]:
            n2["name"] = center_uid

        # ✅ 保留所有边的其他属性
        edge_data = {k: v for k, v in edge.items() if k not in ("node_1", "node_2")}

        flat_triples.append({
            "node_1": n1,
            "node_2": n2,
            **edge_data  # ✅ 展开其它边属性
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
                "relationship": "same_entity"
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

    # 先画非 same_entity 边，单向箭头
    non_same_entity_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("orig_rel") != "same_entity"]
    nx.draw_networkx_edges(G, pos, edgelist=non_same_entity_edges, arrowstyle='-|>', arrowsize=40, edge_color='gray', ax=ax)

    # 再画 same_entity 边，双向箭头
    same_entity_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("orig_rel") == "same_entity"]
    nx.draw_networkx_edges(G, pos, edgelist=same_entity_edges, arrowstyle='<->', arrowsize=45, edge_color='red', width=2, ax=ax)

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
    output_file = "./dual_star_graph/dual_star_structures_v2.json"
    output_dir = "./dual_star_graph"
    os.makedirs(output_dir, exist_ok=True)
    json_output_dir = "./dual_star_json"
    os.makedirs(json_output_dir, exist_ok=True)  # 新增目录保存 JSON 文件

    # Step 1: 加载所有图三元组
    triples = load_graphs(input_files)

    # Step 2: 提取所有实体名的出现位置（记录每个实体出现在哪些三元组中）
    entity_occurrences = defaultdict(list)
    # print("Sample triple:", triples[0])
    # print("Type of triple[0]:", type(triples[0]))
    for idx, triple in enumerate(triples):
        for node in [triple["node_1"], triple["node_2"]]:
            entity_occurrences[node["name"]].append(idx)

    # Step 3: 找出所有重复实体（同一实体名出现两次以上）
    repeated_entities = {ename: idx_list for ename, idx_list in entity_occurrences.items() if len(idx_list) >= 2}
    print(f"[INFO] 发现 {len(repeated_entities)} 个重复实体")

    # Step 4: 为每个重复实体构造多个星型结构（每个三元组为一个局部星）
    entity_stars = {}
    for entity_name, triple_indices in repeated_entities.items():
        stars = []
        for i, idx in enumerate(triple_indices):
            triple = triples[idx]
            local_adj = build_adjacency_outgoing([triple])  # 只构建中心节点出边
            star = build_star_structure_outgoing(entity_name, local_adj, [triple])
            stars.append(star)
        entity_stars[entity_name] = stars

    # Step 5: 构造双星结构并可视化（选择外接实体最多的前 k 个组合）
    all_flat_triples = []
    star_uid_mapping = []
    MAX_DUAL_STARS_PER_ENTITY = 5  # 👈 每个重复的中心最多构造k个双星图

    for entity_name, star_list in entity_stars.items():
        pairs_with_scores = []

        # 枚举所有 (i, j) 组合
        for i in range(len(star_list)):
            for j in range(i + 1, len(star_list)):
                star_i_neighbors = set(n["name"] for n in star_list[i]["neighbors"])
                star_j_neighbors = set(n["name"] for n in star_list[j]["neighbors"])
                union_neighbors = star_i_neighbors.union(star_j_neighbors)

                # 过滤掉中心实体本身
                union_neighbors.discard(entity_name)

                score = len(union_neighbors)  # 👈 连接实体数量
                pairs_with_scores.append(((i, j), score))

        # 按照外接实体数量排序
        top_pairs = sorted(pairs_with_scores, key=lambda x: x[1], reverse=True)[:MAX_DUAL_STARS_PER_ENTITY]

        for (i, j), score in top_pairs:
            uid1 = f"{entity_name}__S{i}"
            uid2 = f"{entity_name}__S{j}"

            flat1 = convert_star_to_flat_triples(star_list[i], unique_prefix=f"S{i}")
            flat2 = convert_star_to_flat_triples(star_list[j], unique_prefix=f"S{j}")

            combined = flat1 + flat2 + [{
                "node_1": {"name": uid1, "label": "VirtualCenter"},
                "node_2": {"name": uid2, "label": "VirtualCenter"},
                "relationship": "same_entity"
            }]

            # 文件名清理（防止非法字符）
            safe_uid1 = uid1.replace(" ", "_").replace("/", "_")
            safe_uid2 = uid2.replace(" ", "_").replace("/", "_")

            # 可视化 PNG 图
            title = f"Dual-Star Structure: {uid1} ↔ {uid2}"
            filename = f"{output_dir}/dual_star_{safe_uid1}_vs_{safe_uid2}.png"
            visualize_graph(combined, llm, title, filename)

            # ✅ 保存为 JSON
            json_filename = f"{json_output_dir}/dual_star_{safe_uid1}_vs_{safe_uid2}.json"
            with open(json_filename, "w", encoding="utf-8") as jf:
                json.dump(combined, jf, indent=2, ensure_ascii=False)

            # 更新记录
            all_flat_triples.extend(combined)
            star_uid_mapping.extend([uid1, uid2])

    # Step 6: 去重UID后，连接所有虚拟中心，构建“same_entity”的总图
    unique_star_uids = list(set(star_uid_mapping))
    virtual_links = generate_virtual_links(unique_star_uids)
    all_flat_triples.extend(virtual_links)

    visualize_graph(
        all_flat_triples,
        llm,
        title="All Dual-Star Structures with Virtual Links",
        output_file=f"{output_dir}/dual_star_virtual_links.png"
    )

    # Step 7: 保存所有星型结构数据（扁平化保存）
    all_stars_flat = []
    for stars in entity_stars.values():
        for star in stars:
            all_stars_flat.append(star)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_stars_flat, f, indent=2, ensure_ascii=False)
    print(f"[DONE] 所有星型结构已保存至 {output_file}")


if __name__ == "__main__":
    main()
