"""
第三种相关关系捕获，提取既不隶属于同一三元组的主语和宾语，也不是相同实体的，只是语义关联的两个实体作为双星结构的两个星型的中心
捕获图中可能的双星结构，其中两个中心构成三元组的连接关系，双星结构的两个星型属于有关联的星型结构
呈现总的双星结构，分开呈现所有的双星结构
sampled_pairs = candidate_pairs[:50]：随机挑选50对实体对进行语义相关性判断
目前可以判断匹配的双星结构的中心节点是否有出边的邻居节点，如果有邻居节点，最多保留两跳之内的邻居
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
import random


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
    """
    将星型结构转换成扁平三元组格式，给中心节点添加唯一标识，且保留边的所有属性。
    """
    flat_triples = []
    center_uid = f"{star['center']}__{unique_prefix}"

    for edge in star["edges"]:
        # 深度复制整条边
        edge_copy = copy.deepcopy(edge)

        # 替换中心节点名称
        if edge_copy["node_1"]["name"] == star["center"]:
            edge_copy["node_1"]["name"] = center_uid
        if edge_copy["node_2"]["name"] == star["center"]:
            edge_copy["node_2"]["name"] = center_uid

        # 保留其他属性，比如 relationship、weight、timestamp 等
        flat_triples.append(edge_copy)

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

    # 再画 related 或 semantic_related 边，双向箭头
    related_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("orig_rel") in {"related", "semantic_related"}
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=related_edges,
        arrowstyle='<->', arrowsize=45, edge_color='red', width=2, ax=ax
    )

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


def build_filtered_star_structure(center, other_center, adjacency, triples):
    """
    构建过滤后的星型结构，只保留与另一个中心相关联的邻居。

    - center: 当前星型结构中心节点名称
    - other_center: 另一个中心节点名称
    - adjacency: 邻接表（中心 -> 邻居列表）
    - triples: 原始三元组列表

    逻辑：
    - 只保留邻居中，有边连接到 other_center 或 other_center 的邻居的节点
    """
    star = {
        "center": center,
        "center_properties": get_node_properties(center, triples),
        "neighbors": [],
        "edges": []
    }

    neighbors = adjacency.get(center, [])

    # 获取 other_center 的邻居，方便判断
    other_neighbors = set([n for n, _ in adjacency.get(other_center, [])])

    for neighbor_name, edge in neighbors:
        # 过滤条件：
        # 1. 邻居是另一个中心节点本身
        # 2. 邻居与另一个中心的邻居有交集（即邻居间可能相连）
        # 3. 或者其他自定义的语义相关判断(可选)

        if neighbor_name == other_center or neighbor_name in other_neighbors:
            star["neighbors"].append(get_node_properties(neighbor_name, triples))
            star["edges"].append(edge)
        else:
            # 可以根据需要添加更多判断条件，比如语义相关度等
            pass

    return star


def build_adjacency_outgoing(triples):
    adj = defaultdict(list)
    for triple in triples:
        n1 = triple["node_1"]["name"]
        n2 = triple["node_2"]["name"]
        adj[n1].append((n2, triple))  # 只添加中心节点指向邻居的边
    return adj


def in_same_triple(e1, e2, triples):
    return any({e1, e2} <= {t["node_1"]["name"], t["node_2"]["name"]} for t in triples)

def is_semantically_related(desc1, desc2, llm):
    prompt = (
        f"Are the following two entities semantically related? Answer 'yes' or 'no'.\n"
        f"1) {desc1}\n2) {desc2}\nIf yes, briefly explain how."
    )
    resp = llm.generate(user_message=prompt, system_message="You are concise.")
    return resp.strip().lower().startswith("yes")

# def is_semantically_related(desc1, desc2, llm):
#     prompt = (
#         "You are a semantic reasoning agent. "
#         "Given the following two entity descriptions, determine whether the entities are semantically related.\n\n"
#         "Two entities are considered semantically related if they refer to the same person, thing, event, organization, or concept, "
#         "or if they play a meaningful role in the same context, activity, or scenario.\n\n"
#         "Please answer with 'Yes' if they are semantically related, or 'No' if not. "
#         "If 'Yes', briefly explain the relationship in one sentence.\n\n"
#         f"Entity 1: {desc1}\n"
#         f"Entity 2: {desc2}\n\n"
#         "Answer:"
#     )
#
#     resp = llm.generate(user_message=prompt, system_message="You are careful, concise, and logical.")
#     print(f"\n❓ Checking: {desc1} ↔ {desc2}")
#     print(f"Response: {resp.strip()}")
#
#     return resp.strip().lower().startswith("yes")


def generate_general_descriptions(json_data):
    from collections import OrderedDict

    nodes = OrderedDict()  # 保持顺序 + 去重

    for entry in json_data:
        for key in ["node_1", "node_2"]:
            node = entry[key]
            name = node.get("name", node.get("label", "Unknown"))

            # 合并属性（后者覆盖前者）
            if name not in nodes:
                nodes[name] = node.copy()
            else:
                nodes[name].update(node)

    descriptions = []

    for name, attrs in nodes.items():
        label = attrs.get("label", "Entity")
        rest_attrs = {k: v for k, v in attrs.items() if k not in {"name", "label"}}

        if rest_attrs:
            props_str = ", ".join(f"{k} is {v}" for k, v in rest_attrs.items())
            sentence = f"{name} is a {label} with the following attributes: {props_str}."
        else:
            sentence = f"{name} is a {label}."
        descriptions.append(sentence)

    return descriptions

def refine_with_llm(sentences, llm, batch_size=1):
    refined = []
    system_prompt = (
        "You are a professional writer and editor. Your task is to rewrite the following sentence "
        "so that it sounds natural, fluent, and contextually appropriate in English. Ensure that:\n"
        "- All factual information is preserved accurately.\n"
        "- The sentence flows smoothly and reads like it was written by a native English speaker.\n"
        "- Attributes such as age, occupation, location, affiliation, and status are clearly and concisely expressed.\n"
        "- Avoid redundancy or awkward phrasing.\n"
        "Return only the revised sentence without any explanations or extra commentary."
    )

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        for sentence in batch:
            try:
                response = llm.generate(
                    user_message=sentence,
                    system_message=system_prompt
                )
                refined.append(response.strip())
            except Exception as e:
                print(f"⚠️ Error processing sentence: {sentence}\n{e}")
                refined.append(sentence)  # fallback to original

    return refined

def load_graphs_by_file(file_list):
    """
    返回 dict：filename -> triples
    """
    all_data = {}
    for fpath in file_list:
        with open(fpath, "r", encoding="utf-8") as f:
            all_data[fpath] = json.load(f)
    return all_data


def build_adjacency_outgoing(triples):
    adj = defaultdict(list)
    for triple in triples:
        n1 = triple["node_1"]["name"]
        n2 = triple["node_2"]["name"]
        adj[n1].append((n2, triple))  # 有向边 n1 -> n2
    return adj


def build_depth2_subgraph_from(center, adjacency, triples):
    """
    构建最大深度为2的出边子图（只向外扩展，路径长度 ≤ 2）
    """
    visited = set()
    edges = []
    neighbors = set()
    queue = [(center, 0)]

    while queue:
        current_node, depth = queue.pop(0)
        if depth > 2:
            continue
        if current_node in visited and depth > 0:  # 根节点可以重复扩展
            continue
        visited.add(current_node)

        for neighbor, edge in adjacency.get(current_node, []):
            edges.append(edge)
            neighbors.add(neighbor)
            if depth < 2:
                queue.append((neighbor, depth + 1))

    star = {
        "center": center,
        "center_properties": get_node_properties(center, triples),
        "neighbors": [get_node_properties(n, triples) for n in neighbors],
        "edges": edges
    }
    return star



def main():
    input_files = ["graph1_enriched.json", "graph2_enriched.json"]
    graphs_by_file = load_graphs_by_file(input_files)
    adjacency_by_file = {
        fname: build_adjacency_outgoing(triples)
        for fname, triples in graphs_by_file.items()
    }
    triples = load_graphs(input_files)
    adjacency = build_adjacency_outgoing(triples)
    os.makedirs("semantic_dual_star_graph", exist_ok=True)
    # 确保输出目录存在
    graph_output_dir = "semantic_dual_star_graph"
    json_output_dir = "semantic_dual_star_json"
    os.makedirs(json_output_dir, exist_ok=True)

    # 1) 生成实体自然语言描述
    basic_descs = generate_general_descriptions(triples)
    # descs = refine_with_llm(basic_descs, llm)
    entity_desc = {s.split(" is ")[0]: s for s in basic_descs}

    # 2) 从所有实体中随机抽取 20 对不同且不属于同一三元组的实体
    entities = list(entity_desc.keys())
    candidate_pairs = [
        (e1, e2)
        for i, e1 in enumerate(entities)
        for j, e2 in enumerate(entities)
        if i < j and e1 != e2 and not in_same_triple(e1, e2, triples)
    ]
    random.shuffle(candidate_pairs)
    # 随机抽取50对实体判断是否相关
    sampled_pairs = candidate_pairs[:20]

    sem_pairs = []
    for e1, e2 in sampled_pairs:
        if is_semantically_related(entity_desc[e1], entity_desc[e2], llm):
            sem_pairs.append((e1, e2))

    print(f"✅ 语义相关的实体对数量: {len(sem_pairs)}")

    all_flat = []
    uids = []

    # 3) 构建并保存每个语义双星结构
    for idx, (e1, e2) in enumerate(sem_pairs):
        # star1 = build_filtered_star_structure(e1, e2, adjacency, triples)
        # star2 = build_filtered_star_structure(e2, e1, adjacency, triples)
        triples1 = graphs_by_file["graph1_enriched.json"]
        triples2 = graphs_by_file["graph2_enriched.json"]
        adj1 = adjacency_by_file["graph1_enriched.json"]
        adj2 = adjacency_by_file["graph2_enriched.json"]
        print(f"[INFO] Pair #{idx}: {e1} ↔ {e2}")
        print(f" - {e1} neighbors in graph1: {len(adj1.get(e1, []))}")
        print(f" - {e2} neighbors in graph2: {len(adj2.get(e2, []))}")

        star1 = build_depth2_subgraph_from(e1, adj1, triples1)
        star2 = build_depth2_subgraph_from(e2, adj2, triples2)

        uid1_raw = e1.replace(" ", "_").replace("/", "_")
        uid2_raw = e2.replace(" ", "_").replace("/", "_")
        uid1 = f"{uid1_raw}__SR{idx}_A"
        uid2 = f"{uid2_raw}__SR{idx}_B"

        f1 = convert_star_to_flat_triples(star1, unique_prefix=f"SR{idx}_A")
        f2 = convert_star_to_flat_triples(star2, unique_prefix=f"SR{idx}_B")

        combined = f1 + f2 + [{
            "node_1": {"name": uid1, "label": "VirtualCenter"},
            "node_2": {"name": uid2, "label": "VirtualCenter"},
            "relationship": "semantically_related"
        }]

        # === 可视化 PNG 图 ===
        title = f"Semantic Dual-Star: {e1} ↔ {e2}"
        png_filename = f"dual_star_{idx}_{uid1}_vs_{uid2}.png"
        png_path = os.path.join(graph_output_dir, png_filename)

        visualize_graph(combined, llm, title=title, output_file=png_path)

        # === 保存 JSON 文件 ===
        json_filename = f"dual_star_{idx}_{uid1}_vs_{uid2}.json"
        json_path = os.path.join(json_output_dir, json_filename)

        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(combined, f_json, indent=2, ensure_ascii=False)

        # 用于后续汇总
        all_flat.extend(combined)
        uids.extend([uid1, uid2])

    # 4) 生成汇总图
    unique_uids = list(set(uids))
    virtual_links = generate_virtual_links(unique_uids)
    all_flat.extend(virtual_links)

    visualize_graph(
        all_flat, llm,
        title="All Semantic Dual-Star Structures",
        output_file="semantic_dual_star_graph/all_semantic_dual_star.png"
    )

    # 输出 JSON 文件
    summary_json_path = "semantic_dual_star_graph/all_semantic_dual_star.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_flat, f, indent=2, ensure_ascii=False)

    print("[DONE] 所有双星结构图已生成并保存到 semantic_dual_star/ 文件夹")

if __name__ == "__main__":
    main()
