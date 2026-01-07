"""
根据生成的属性图json文件可视化属性图
将边的属性呈现在边上
将节点属性呈现在节点旁边，可能会导致节点属性文本框和边的属性文本覆盖
useful
"""

import json
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy


def visualize_property_graph_with_attrs(json_data):
    G = nx.DiGraph()

    label_colors = {
        "Person": "#87CEFA",
        "Company": "#FFD700",
        "Event": "#FFB6C1",
        "Department": "#98FB98",
        "Location": "#DDA0DD",
        "default": "#D3D3D3"
    }

    for entry in json_data:
        node1 = copy.deepcopy(entry["node_1"])
        node2 = copy.deepcopy(entry["node_2"])
        rel = entry["relationship"]
        edge_props = entry.get("edge_properties", {})

        name1 = node1.pop("name", node1.pop("label", "Unknown"))
        name2 = node2.pop("name", node2.pop("label", "Unknown"))
        label1 = node1.pop("label", "default")
        label2 = node2.pop("label", "default")

        G.add_node(name1, label=label1, **node1)
        G.add_node(name2, label=label2, **node2)

        edge_label = rel
        if edge_props:
            props = "\n".join(f"{k}: {v}" for k, v in edge_props.items())
            edge_label += f"\n[{props}]"

        G.add_edge(name1, name2, label=edge_label)

    pos = nx.spring_layout(G, k=1.2, iterations=100)

    plt.figure(figsize=(18, 14))
    node_colors = [label_colors.get(G.nodes[n].get("label", "default"), "#D3D3D3") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=2200, node_color=node_colors, edgecolors="black")

    node_labels = {node: f"{node}\n({G.nodes[node].get('label','Entity')})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=50, edge_color='gray')

    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', rotate=False, label_pos=0.5)

    # 在节点旁边绘制属性文本框
    ax = plt.gca()
    for node, (x, y) in pos.items():
        attrs = G.nodes[node].copy()
        label = attrs.pop("label", None)
        # 去除主显示标签，只显示额外属性
        if attrs:
            # 拼接属性文本
            attr_text = "\n".join(f"{k}: {v}" for k, v in attrs.items())
            # 调整文本框位置（稍微右上偏移）
            ax.text(x + 0.05, y + 0.05, attr_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    plt.title("Enhanced Knowledge Graph Visualization with Node Attributes", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("knowledge_graph.png", dpi=300)  # 保存为png，300dpi清晰度
    plt.show()


# 载入数据并可视化
with open("output_knowledge_graph.json", "r", encoding="utf-8") as f:
    data = json.load(f)

visualize_property_graph_with_attrs(data)
