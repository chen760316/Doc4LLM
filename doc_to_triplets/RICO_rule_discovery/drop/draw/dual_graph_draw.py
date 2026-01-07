"""
绘制双星图
图中存在交叉的边，并且部分边的标签长度大于边的长度
"""

import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和黑白主题
plt.rcParams["font.family"] = "Arial"

def offset_path_nodes(pos, path, offset_max=0.1):
    n = len(path)
    for i, node in enumerate(path):
        if node in pos:
            # 偏移大小按照距离源节点远近成反比，越靠近源节点偏移越大
            weight = (n - i) / n
            offset = offset_max * weight
            pos[node] = (pos[node][0], pos[node][1] + ((-1) ** i) * offset)
    return pos


def draw_query_graph(Q_name, edges, edge_labels, node_types, pos=None, subplot_index=1):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    ax = plt.subplot(1, 5, subplot_index)
    if not pos:
        pos = nx.spring_layout(G, seed=42)

        # 找入度为0的中心节点（源节点），这里假设 x0 和 y0
        sources = [n for n, d in G.in_degree if d == 0]

        # 分别处理 x0 和 y0 星型结构
        for source in sources:
            # 找所有邻接节点（第一层）
            neighbors = list(G.successors(source))

            # 计算每个邻居的新坐标
            # 给x0分配上半圆角度 0~180°
            # 给y0分配下半圆角度 180~360°
            if source.startswith('x'):
                angles = np.linspace(np.pi, 0, len(neighbors))  # 上半圆，从左180°到右0°
                radius = 0.3 + 0.1 * len(neighbors)  # 半径可调节，控制距离中心节点的远近
            elif source.startswith('y'):
                angles = np.linspace(np.pi, 2 * np.pi, len(neighbors))  # 下半圆，从左180°到右360°
                radius = 0.3 + 0.1 * len(neighbors)

            # 中心节点保持原位置
            center_x, center_y = pos[source]

            for angle, nbr in zip(angles, neighbors):
                # 极坐标转笛卡尔坐标
                new_x = center_x + radius * np.cos(angle)
                new_y = center_y + radius * np.sin(angle)
                pos[nbr] = (new_x, new_y)

    labels = {
        n: f"{n}\n({node_types[n]})" if node_types.get(n) else n
        for n in G.nodes()
    }

    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="->",
        width=1.2,
        connectionstyle='arc3,rad=0.0',
        edge_color='black'
    )

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_color="black"
    )

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="black",
        label_pos=0.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
    )

    plt.title(Q_name, fontsize=12, y=-0.15)
    plt.axis('off')


# 定义 Query Graph 结构
queries = {
    "Q1": {
        "edges": [
            ("x0", "x1"), ("x0", "x2"), ("x0", "x3"), ("x0", "x4"),  # 原来的x星型
            ("y0", "y1"), ("y0", "y2"), ("y0", "y3"), ("y0", "y4")  # 新增的y星型
        ],
        "edge_labels": {
            ("x0", "x1"): "auther",
            ("x0", "x2"): "title",
            ("x0", "x3"): "platform",
            ("x0", "x4"): "version",
            ("y0", "y1"): "auther",
            ("y0", "y2"): "title",
            ("y0", "y3"): "platform",
            ("y0", "y4"): "version",
        },
        "node_types": {
            "x0": "Video", "x1": "Person", "x2": "", "x3": "", "x4": "",
            "y0": "Video", "y1": "Person", "y2": "", "y3": "", "y4": ""
        }
    },
    "Q2": {
        "edges": [
            ("x0", "x1"), ("x0", "x2"), ("x2", "x3"), ("x0", "x4"), ("x0", "x5"),
            ("y0", "y1"), ("y0", "y2"), ("y2", "y3"), ("y0", "y4"), ("y0", "y5"), ("y3", "y6")
        ],
        "edge_labels": {
            ("x0", "x1"): "model",
            ("x0", "x2"): "country",
            ("x2", "x3"): "state",
            ("x0", "x4"): "connected_to",
            ("x0", "x5"): "status",
            ("y0", "y1"): "model",
            ("y0", "y2"): "country",
            ("y2", "y3"): "state",
            ("y0", "y4"): "connected_to",
            ("y0", "y5"): "status",
            ("y3", "y6"): "city"
        },
        "node_types": {
            "x0": "Product", "x1": "", "x2": "", "x3": "", "x4": "", "x5": "",
            "y0": "Product", "y1": "", "y2": "", "y3": "", "y4": "", "y5": "", "y6": ""
        }
    },
    "Q3": {
        "edges": [
            ("x0", "x1"), ("x1", "x2"), ("x1", "x3"), ("x1", "x5"), ("x2", "x4"),
            ("y0", "y1"), ("y1", "y2"), ("y1", "y3"), ("y1", "y5"), ("y2", "y4")
        ],
        "edge_labels": {
            ("x0", "x1"): "write",
            ("x1", "x2"): "description",
            ("x1", "x3"): "commented_by",
            ("x1", "x5"): "reader",
            ("x2", "x4"): "to_string",
            ("y0", "y1"): "write",
            ("y1", "y2"): "description",
            ("y1", "y3"): "commented_by",
            ("y1", "y5"): "reader",
            ("y2", "y4"): "to_string",
        },
        "node_types": {
            "x0": "Person", "x1": "Paper", "x2": "", "x3": "", "x4": "", "x5": "",
            "y0": "Person", "y1": "Paper", "y2": "", "y3": "", "y4": "", "y5": ""
        }
    },
    "Q4": {
        "edges": [
            ("x0", "x1"), ("x0", "x2"), ("x0", "x5"), ("x0", "x6"), ("x0", "x7"), ("x2", "x3"), ("x3", "x4"),
            ("y0", "y1"), ("y0", "y2"), ("y0", "y5"), ("y0", "y6"), ("y0", "y7"), ("y2", "y3"), ("y3", "y4"),
        ],
        "edge_labels": {
            ("x0", "x1"): "brand",
            ("x0", "x2"): "state",
            ("x0", "x5"): "open_date",
            ("x0", "x6"): "supplied_by",
            ("x0", "x7"): "revenue",
            ("x2", "x3"): "city",
            ("x3", "x4"): "detailed_address",
            ("y0", "y1"): "brand",
            ("y0", "y2"): "state",
            ("y0", "y5"): "open_date",
            ("y0", "y6"): "supplied_by",
            ("y0", "y7"): "revenue",
            ("y2", "y3"): "city",
            ("y3", "y4"): "detailed_address",
        },
        "node_types": {
            "x0": "shop", "x1": "", "x2": "", "x3": "", "x4": "", "x5": "", "x6": "", "x7": "",
            "y0": "shop", "y1": "", "y2": "", "y3": "", "y4": "", "y5": "", "y6": "", "y7": ""
        }
    },
    "Q5": {
        "edges": [
            ("x0", "x1"), ("x0", "x2"), ("x0", "x3"), ("x2", "x4"),
            ("y0", "y1"), ("y0", "y2"), ("y0", "y3")
        ],
        "edge_labels": {
            ("x0", "x1"): "creator",
            ("x0", "x2"): "exhibited_at",
            ("x0", "x3"): "description",
            ("x2", "x4"): "located_in",
            ("y0", "y1"): "creator",
            ("y0", "y2"): "exhibited_at",
            ("y0", "y3"): "description",
        },
        "node_types": {
            "x0": "Product", "x1": "", "x2": "", "x3": "", "x4": "",
            "y0": "Product", "y1": "", "y2": "", "y3": ""
        }
    }
}

# 创建图
plt.figure(figsize=(20, 5))
for i, (qname, qdata) in enumerate(queries.items(), 1):
    draw_query_graph(
        Q_name=qname,
        edges=qdata["edges"],
        edge_labels=qdata["edge_labels"],
        node_types=qdata["node_types"],
        subplot_index=i
    )

plt.tight_layout()
plt.savefig("dual_graphs.svg", format="svg")  # 保存为SVG文件
plt.show()
