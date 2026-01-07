"""
绘制双星图
"""

import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import textwrap

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
        pos = {}
        center_gap_y = 7.5  # 控制上下星型的距离

        # 布局中心节点：x0 在上，y0 在下
        pos['x0'] = (0, center_gap_y / 2)
        pos['y0'] = (0, -center_gap_y / 2)

        # 设置子节点布局角度
        for center_node, base_angle in [('x0', 270), ('y0', 90)]:
            neighbors = list(G.successors(center_node))
            n = len(neighbors)
            angle_range = 180  # 展开角度
            start_angle = base_angle - angle_range / 2

            radius = 3.0 # 控制边的长度
            cx, cy = pos[center_node]

            for i, nbr in enumerate(neighbors):
                angle_deg = start_angle + i * (angle_range / max(n - 1, 1))
                angle_rad = np.deg2rad(angle_deg)

                dx = radius * np.cos(angle_rad)
                dy = radius * np.sin(angle_rad)

                pos[nbr] = (cx + dx, cy + dy)

        # 自动放置非 x0/y0 扇形结构下的其他路径节点
        for (src, tgt) in edges:
            if tgt not in pos and src in pos:
                sx, sy = pos[src]
                pos[tgt] = (sx + 2.0, sy - 1.2)

    # 节点标签格式
    labels = {
        n: f"{n}\n({node_types[n]})" if node_types.get(n) else n
        for n in G.nodes()
    }

    # 包装边标签，防止过长
    def wrap_label(label):
        return "\n".join(textwrap.wrap(label, width=20))

    formatted_edge_labels = {
        (u, v): wrap_label(lbl) for (u, v), lbl in edge_labels.items()
    }

    # 绘图
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="->",
        width=1.4,
        connectionstyle='arc3,rad=0.0',
        edge_color='black'
    )

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=9,  # 默认字号为9
        font_color="black"
    )

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=formatted_edge_labels,
        font_size=8,   # 默认字号为8
        font_color="black",
        label_pos=0.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
    )

    plt.title(Q_name, fontsize=12, y=-0.01)  # 标题在子图下方，但距离子图近一点
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
            ("x0", "x1"), ("x0", "x2"), ("x0", "x3"), ("x2", "x4"), ("x0", "x5"),
            ("y0", "y1"), ("y0", "y2"), ("y0", "y3"), ("y2", "y4"), ("y0", "y5")
        ],
        "edge_labels": {
            ("x0", "x1"): "writer",
            ("x0", "x2"): "description",
            ("x0", "x3"): "commented_by",
            ("x0", "x5"): "reader",
            ("x2", "x4"): "to_string",
            ("y0", "y1"): "writer",
            ("y0", "y2"): "description",
            ("y0", "y3"): "commented_by",
            ("y0", "y5"): "reader",
            ("y2", "y4"): "to_string",
        },
        "node_types": {
            "x0": "Paper", "x1": "Person", "x2": "", "x3": "", "x4": "", "x5": "",
            "y0": "Paper", "y1": "Person", "y2": "", "y3": "", "y4": "", "y5": ""
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
            ("x3", "x4"): "address",
            ("y0", "y1"): "brand",
            ("y0", "y2"): "state",
            ("y0", "y5"): "open_date",
            ("y0", "y6"): "supplied_by",
            ("y0", "y7"): "revenue",
            ("y2", "y3"): "city",
            ("y3", "y4"): "address",
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

# 调整整体间距，减少子图左右间距和水平间距
plt.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.1, wspace=0.01)

for i, (qname, qdata) in enumerate(queries.items(), 1):
    draw_query_graph(
        Q_name=qname,
        edges=qdata["edges"],
        edge_labels=qdata["edge_labels"],
        node_types=qdata["node_types"],
        subplot_index=i
    )

fig = plt.gcf()
# 添加竖线分割，总共4条竖线，放在子图之间
for i in range(1, 5):
    # 子图的宽度 = (right - left) / 5, 这里left=0.03, right=0.97, 宽度约为0.188
    left = 0.03
    width = (0.97 - 0.03) / 5
    x = left + i * width  # 竖线x坐标，放在子图间隔处

    # 添加细长Axes绘制竖线
    ax_line = fig.add_axes([x - 0.001, 0.1, 0.002, 0.8])  # [left, bottom, width, height]，高度和主图差不多
    ax_line.axvline(0.5, color='gray', linestyle='--', linewidth=1)
    ax_line.axis('off')  # 关闭坐标轴显示

plt.savefig("dual_graphs_with_lines_initial.pdf", format="pdf")
plt.show()


