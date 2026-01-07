"""
根据生成的属性图json文件可视化属性图
将边的属性呈现在边上
将节点属性呈现在侧边栏
"""

import json
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy


def visualize_property_graph_with_attrs_sidepanel(json_data):
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

    fig, ax = plt.subplots(figsize=(20, 14))

    node_colors = [label_colors.get(G.nodes[n].get("label", "default"), "#D3D3D3") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=2200, node_color=node_colors, edgecolors="black", ax=ax)

    node_labels = {node: f"{node}\n({G.nodes[node].get('label','Entity')})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=50, edge_color='gray', ax=ax)

    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', rotate=False, label_pos=0.5, ax=ax)

    # 收集所有节点属性文本，放到右侧空白区域
    attr_texts = []
    for node in G.nodes():
        attrs = G.nodes[node].copy()
        label = attrs.pop("label", None)
        if attrs:
            text = f"{node} Attributes:\n" + "\n".join(f"  {k}: {v}" for k, v in attrs.items())
            attr_texts.append(text)

    full_attr_text = "\n\n".join(attr_texts)

    # 在图的右侧添加文本框显示所有节点属性
    plt.subplots_adjust(right=0.7)
    ax.text(1.02, 0.5, full_attr_text, fontsize=10, va='center', ha='left', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.title("Enhanced Knowledge Graph Visualization with Node Attributes", fontsize=18)
    plt.axis("off")
    plt.savefig("graph2.png", dpi=300)  # 保存为png，300dpi清晰度
    plt.show()



# 载入数据并可视化
with open("graph2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

visualize_property_graph_with_attrs_sidepanel(data)
