"""
图连接后得到merged_graph.json和merged_graph.png两个文件
useful
"""
import json
import copy
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def build_graph(json_data, prefix=""):
    G = nx.DiGraph()
    name_map = {}

    for entry in json_data:
        node1 = copy.deepcopy(entry["node_1"])
        node2 = copy.deepcopy(entry["node_2"])
        rel = entry["relationship"]
        edge_props = entry.get("edge_properties", {})

        name1 = node1.pop("name", node1.pop("label", "Unknown"))
        name2 = node2.pop("name", node2.pop("label", "Unknown"))
        label1 = node1.pop("label", "default")
        label2 = node2.pop("label", "default")

        prefixed_name1 = f"{prefix}:{name1}"
        prefixed_name2 = f"{prefix}:{name2}"

        name_map[name1] = prefixed_name1
        name_map[name2] = prefixed_name2

        G.add_node(prefixed_name1, label=label1, original_name=name1, **node1)
        G.add_node(prefixed_name2, label=label2, original_name=name2, **node2)

        edge_label = rel
        if edge_props:
            props = "\n".join(f"{k}: {v}" for k, v in edge_props.items())
            edge_label += f"\n[{props}]"

        G.add_edge(prefixed_name1, prefixed_name2, label=edge_label)

    return G, name_map

def add_cross_graph_connections(G, name_map1, name_map2):
    for original_name in name_map1:
        if original_name in name_map2:
            node1 = name_map1[original_name]
            node2 = name_map2[original_name]
            G.add_edge(node1, node2, label="Connected", style="dashed", color="blue")

def visualize_combined_graph(G):
    pos = nx.spring_layout(G, k=1.5)

    fig, ax = plt.subplots(figsize=(22, 16))

    label_colors = {
        "Person": "#87CEFA",
        "Company": "#FFD700",
        "Event": "#FFB6C1",
        "Department": "#98FB98",
        "Location": "#DDA0DD",
        "default": "#D3D3D3"
    }

    # 设置节点颜色
    node_colors = []
    for node in G.nodes():
        label = G.nodes[node].get("label", "default")
        color = label_colors.get(label, "#D3D3D3")
        node_colors.append(color)
    # 调节节点大小的参数node_size
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, edgecolors="black", ax=ax)

    # 设置节点标签
    node_labels = {
        node: f"{G.nodes[node].get('original_name')}\n({G.nodes[node].get('label')})"
        for node in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)

    # 画边
    for u, v, d in G.edges(data=True):
        style = d.get('style', 'solid')
        color = d.get('color', 'gray')
        # 调节箭头大小的参数arrowsize
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            arrowstyle='-|>', arrowsize=50,
            style=style, edge_color=color, ax=ax
        )

    # 设置边标签（包含完整属性）
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        label = d.get('label', '')
        other_props = {k: v for k, v in d.items() if k != 'label'}
        if other_props:
            props_str = ", ".join(f"{k}: {v}" for k, v in other_props.items())
            label = f"{label}\n[{props_str}]"
        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', ax=ax)

    # 右侧显示所有节点属性文本
    attr_texts = []
    for node in G.nodes():
        attrs = G.nodes[node].copy()
        label = attrs.pop("label", None)
        name = attrs.pop("original_name", node)
        if attrs:
            text = f"{name} Attributes:\n" + "\n".join(f"  {k}: {v}" for k, v in attrs.items())
            attr_texts.append(text)

    full_attr_text = "\n\n".join(attr_texts)

    plt.subplots_adjust(right=0.7)  # 为右侧文本预留空间
    ax.text(1.02, 0.5, full_attr_text, fontsize=10, va='center', ha='left', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.title("Connected Knowledge Graphs with Full Attributes", fontsize=18)
    plt.axis("off")
    plt.savefig("merged_graph.png", dpi=300)
    plt.show()


def main():
    # 读取两个 JSON 文件
    with open("graph1.json", "r", encoding="utf-8") as f1:
        json_data_1 = json.load(f1)

    with open("graph2.json", "r", encoding="utf-8") as f2:
        json_data_2 = json.load(f2)

    # 构建两个图
    G1, map1 = build_graph(json_data_1, prefix="G1")
    G2, map2 = build_graph(json_data_2, prefix="G2")

    # 合并图
    G_combined = nx.compose(G1, G2)

    # 添加跨图连接边
    add_cross_graph_connections(G_combined, map1, map2)

    # 可视化
    visualize_combined_graph(G_combined)

    # 导出为 JSON（包括连接边，且保留所有属性）
    merged_triples = []

    for u, v, data in G_combined.edges(data=True):
        node1 = G_combined.nodes[u]
        node2 = G_combined.nodes[v]

        # 将节点属性全部复制（包含 name 和 label）
        node1_data = {
            "name": node1.get("original_name", u),
            **{k: v for k, v in node1.items() if k != "original_name"}
        }

        node2_data = {
            "name": node2.get("original_name", v),
            **{k: v for k, v in node2.items() if k != "original_name"}
        }

        # 将边的所有属性完整加入 edge_properties
        edge_props = {k: v for k, v in data.items() if k != "label"}

        triple = {
            "node_1": node1_data,
            "node_2": node2_data,
            "relationship": data.get("label", "related_to")
        }

        if edge_props:
            triple["edge_properties"] = edge_props

        merged_triples.append(triple)

    # 写入 JSON 文件
    with open("merged_graph.json", "w", encoding="utf-8") as out_file:
        json.dump(merged_triples, out_file, ensure_ascii=False, indent=4)

    print("Merged graph JSON written to merged_graph.json")


if __name__ == "__main__":
    main()
