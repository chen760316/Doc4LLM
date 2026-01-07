"""
star_discovery_to_star的延续
将G1和G2中(文件夹graph1和graph2下)得到的各个星型结构中心进行匹配，匹配成功的G1和G2的两个星型结构构造双星结构
双星结构的每个星型保留，然后在两个星型结构的中心加入匹配边
"""

import os
import json
# 保存双星结构图像和JSON
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import copy

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

def add_suffix_to_center(triples, center, suffix):
    updated = []
    new_center_name = f"{center}__{suffix}"
    for triple in triples:
        t = copy.deepcopy(triple)
        if t["node_1"]["name"] == center:
            t["node_1"]["name"] = new_center_name
        if t["node_2"]["name"] == center:
            t["node_2"]["name"] = new_center_name
        updated.append(t)
    return updated

# 加载星型结构（扁平化三元组）
def load_star_structures_from_dir(dir_path):
    star_structures = {}
    for fname in os.listdir(dir_path):
        if fname.endswith(".json"):
            path = os.path.join(dir_path, fname)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    triples = json.load(f)
                    if triples:
                        center = triples[0]['node_1']['name']
                        star_structures[center] = triples
                except Exception as e:
                    print(f"⚠️ 文件解析失败: {fname} - {e}")
    return star_structures

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

    print("=== 三元组关系映射 ===")
    for orig, simp in rel_map.items():
        print(f"{orig}  -->  {simp}")
        print("*" * 100)
    print("====================\n")

    # 构建图
    G = nx.DiGraph()
    label_colors = {
        "Person": "#87CEFA", "Company": "#FFD700", "Event": "#FFB6C1",
        "Department": "#98FB98", "Location": "#DDA0DD", "VirtualCenter": "#FF6347",
        "default": "#D3D3D3"
    }

    # 添加节点和边
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

    # 图布局
    pos = nx.spring_layout(G, k=1.2, seed=42)
    fig, ax = plt.subplots(figsize=(20, 14))

    # 绘制节点
    node_colors = [label_colors.get(G.nodes[n].get("label", "default"), "#D3D3D3") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, edgecolors="black", ax=ax)

    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)

    # 分类绘制边
    same_entity_edges = []
    other_edges = []

    for u, v, d in G.edges(data=True):
        if d.get("orig_rel") == "same_entity":
            same_entity_edges.append((u, v))
        else:
            other_edges.append((u, v))

    # 绘制非 same_entity 的边（灰色单向箭头）
    nx.draw_networkx_edges(
        G, pos,
        edgelist=other_edges,
        arrowstyle='-|>',
        arrowsize=40,
        edge_color='gray',
        style='solid',
        ax=ax
    )

    # 绘制 same_entity 边（红色实线双向箭头）
    nx.draw_networkx_edges(
        G, pos,
        edgelist=same_entity_edges,
        arrowstyle='<->',
        arrowsize=45,
        edge_color='red',
        width=2,
        style='solid',
        ax=ax
    )

    # 添加边标签
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', ax=ax)

    # 图标题与保存
    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[SAVED] 图已保存为: {output_file}")


# 加载 graph1 和 graph2 中的星型图
graph1_stars = load_star_structures_from_dir("./star_json/graph1")
graph2_stars = load_star_structures_from_dir("./star_json/graph2")

# 找出共同中心节点
common_centers = set(graph1_stars.keys()) & set(graph2_stars.keys())
print(f"[INFO] 发现 {len(common_centers)} 个共同中心实体: {list(common_centers)}")

os.makedirs("./star_graph/dual_graph", exist_ok=True)
os.makedirs("./star_json/dual_graph", exist_ok=True)

for idx, center in enumerate(common_centers, 1):
    # 分别转换两个图的星型结构，加后缀防止融合
    g1_triples = add_suffix_to_center(graph1_stars[center], center, "G1")
    g2_triples = add_suffix_to_center(graph2_stars[center], center, "G2")

    # 构造虚拟连接边：same_entity
    virtual_link = {
        "node_1": {"name": f"{center}__G1", "label": "VirtualCenter"},
        "node_2": {"name": f"{center}__G2", "label": "VirtualCenter"},
        "relationship": "same_entity"
    }

    # 合并两部分 + 虚拟边
    combined_triples = g1_triples + g2_triples + [virtual_link]

    # 安全文件名
    safe_center = center.replace(" ", "_").replace("/", "_")
    output_image = f"./star_graph/dual_graph/dual_star_graph_{idx}_{safe_center}.png"
    output_json = f"./star_json/dual_graph/dual_star_graph_{idx}_{safe_center}.json"

    # 保存 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(combined_triples, f, indent=2, ensure_ascii=False)

    # 可视化
    visualize_property_graph_with_simplified_edges(
        combined_triples,
        llm,
        title=f"Dual Star Structure: {center}",
        output_file=output_image
    )

print("[DONE] 所有双星结构已生成完毕。")