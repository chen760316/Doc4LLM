"""
绘制双星图
用红线标注缺失边
"""

import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

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

# 自定义绘制旋转边标签
def draw_rotated_edge_labels(G, pos, edge_labels, ax, font_size=12, offset=0.35):
    ax.set_aspect('equal')  # 先确保比例

    for (u, v), label in edge_labels.items():
        if u not in pos or v not in pos:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        dx /= norm
        dy /= norm

        lx = mx - dy * offset
        ly = my + dx * offset

        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        ax.text(
            lx, ly, label,
            fontsize=font_size,
            rotation=angle,
            rotation_mode='anchor',  # 尝试改成anchor
            ha='center',
            va='center'
        )


def draw_precise_rotated_edge_labels(G, pos, edge_labels, ax, font_size=12, offset=0.3):
    for (u, v), label in edge_labels.items():
        if u not in pos or v not in pos:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # 边中点
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        # 向量差
        dx = x2 - x1
        dy = y2 - y1

        # 边方向和法向量
        norm = np.hypot(dx, dy)
        if norm == 0:
            continue
        ux, uy = dx / norm, dy / norm  # 单位方向向量
        nx_, ny_ = -uy, ux  # 法向量（朝外）

        # 标签位置（沿法向量偏移）
        lx = mx + offset * nx_
        ly = my + offset * ny_

        # 角度（弧度）
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        if angle_deg > 90:
            angle_deg -= 180
            angle_rad -= np.pi
        elif angle_deg < -90:
            angle_deg += 180
            angle_rad += np.pi

        # 创建 TextPath
        text_path = TextPath((0, 0), label, size=font_size)

        # 变换：旋转 + 平移
        trans = Affine2D().rotate(angle_rad).translate(lx, ly) + ax.transData

        # 添加 PathPatch 到 Axes
        patch = PathPatch(text_path, transform=trans, color="black", lw=0)
        ax.add_patch(patch)


def draw_node_labels_custom(ax, G, pos, labels, center_nodes):
    for node, label in labels.items():
        x, y = pos[node]

        # 拆分 label 成 name 和 node_type
        if "\n" in label:
            name, node_type = label.split("\n")
        else:
            name = label
            node_type = ""

        # 将 x0、y1 等转为 $x_0$ 格式
        if name.startswith("x") or name.startswith("y"):
            base = name[0]
            subscript = name[1:]
            if node in center_nodes:
                # 中心节点加粗变量名
                formatted_name = rf"$\mathbf{{{base}_{{{subscript}}}}}$"
            else:
                formatted_name = rf"${base}_{{{subscript}}}$"
        else:
            formatted_name = name

        # 设置偏移量
        if node == center_nodes[0]:  # x0
            offset_x = -0.5
            offset_y = 1
        elif node == center_nodes[1]:  # y0
            offset_x = 0.2
            offset_y = -0.4
        else:
            offset_x = 0
            offset_y = -0.2

        # 拼接最终显示文本
        if node_type:
            # 用原始字符串，并在数学内容外部用 \n 分隔普通文本
            display_text = rf"{formatted_name}" + f"\n{node_type}"
        else:
            display_text = formatted_name

        ax.text(
            x + offset_x, y + offset_y,
            display_text,
            fontsize=12,
            ha='center',
            va='center'
        )




def draw_query_graph(Q_name, edges, edge_labels, node_types, pos=None, subplot_index=1):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    ax = plt.subplot(1, 5, subplot_index)

    if not pos:
        pos = {}

        # 单独增大 Q4 的上下间距
        if Q_name == "Q4":
            center_gap_y = 20.0  # Q4 的间距更大
        else:
            center_gap_y = 13.5  # 默认值

        # 布局中心节点：x0 在上，y0 在下
        pos['x0'] = (0, center_gap_y / 2)
        pos['y0'] = (0, -center_gap_y / 2)

        # 设置子节点布局角度
        for center_node, base_angle in [('x0', 270), ('y0', 90)]:
            neighbors = list(G.successors(center_node))
            n = len(neighbors)
            angle_range = 180  # 展开角度
            start_angle = base_angle - angle_range / 2

            if Q_name == "Q4":
                radius = 7.8  # Q4 的间距更大
            else:
                radius = 6.0  # 默认值

            cx, cy = pos[center_node]

            for i, nbr in enumerate(neighbors):
                angle_deg = start_angle + i * (angle_range / max(n - 1, 1))
                angle_rad = np.deg2rad(angle_deg)

                dx = radius * np.cos(angle_rad)
                dy = radius * np.sin(angle_rad)

                pos[nbr] = (cx + dx, cy + dy)

            # 手动调整 Q4 中 x5 的 y 位置，往上提
            if Q_name == "Q4" and 'x5' in pos:
                x5x, x5y = pos['x5']
                pos['x5'] = (x5x, x5y + 0.3)  # 上调 1.2 个单位

        # 自动放置非 x0/y0 扇形结构下的其他路径节点
        for (src, tgt) in edges:
            if tgt not in pos and src in pos:
                sx, sy = pos[src]

                if Q_name == "Q4":
                    # 增大 Q4 的后续边的长度和垂直偏移
                    offset_x = 3.0
                    offset_y = -2.0
                else:
                    offset_x = 3.0
                    offset_y = -2.2

                pos[tgt] = (sx + offset_x, sy + offset_y)

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

    # 默认所有边颜色为黑色
    edge_colors = ['black'] * len(edges)

    # 如果是 Q5，设置 y0->y2 为红色
    if Q_name == "Q5":
        for i, (u, v) in enumerate(edges):
            if (u, v) == ("y0", "y2"):
                edge_colors[i] = 'red'

    # 绘图时传入 edge_colors
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle="->",
        width=1.4,
        connectionstyle='arc3,rad=0.0',
        edge_color=edge_colors
    )

    draw_node_labels_custom(ax, G, pos, labels, center_nodes=['x0', 'y0'])

    draw_rotated_edge_labels(G, pos, formatted_edge_labels, ax, offset=0.45)
    # draw_precise_rotated_edge_labels(G, pos, formatted_edge_labels, ax, font_size=12, offset=0.3)

    # plt.title(Q_name, fontsize=12, y=-0.05)  # 标题在子图下方，但距离子图近一点
    fig = plt.gcf()
    num_subplots = 5

    for i, qname in enumerate(queries.keys()):
        # 计算标题的 x 位置（均匀分布在0~1之间）
        x = (i + 0.5) / num_subplots
        y = 0.05
        # 转换 Q1 -> $Q_1$ 等
        latex_label = rf"$Q_{{{qname[1:]}}}$"
        fig.text(x, y, latex_label, ha='center', va='bottom', fontsize=12)

    plt.axis('off')






# 定义 Query Graph 结构
queries = {
    "Q1": {
        "edges": [
            ("x0", "x1"), ("x0", "x2"), ("x0", "x3"), ("x0", "x4"),  # 原来的x星型
            ("y0", "y1"), ("y0", "y2"), ("y0", "y3"), ("y0", "y4")  # 新增的y星型
        ],
        "edge_labels": {
            ("x0", "x1"): "creator",
            ("x0", "x2"): "title",
            ("x0", "x3"): "platform",
            ("x0", "x4"): "version",
            ("y0", "y1"): "creator",
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
            ("x0", "x1"), ("x0", "x2"), ("x2", "x3"), ("x0", "x4"),
            ("y0", "y1"), ("y0", "y2"), ("y2", "y3"), ("y0", "y4"), ("y3", "y5")
        ],
        "edge_labels": {
            ("x0", "x1"): "model",
            ("x0", "x2"): "country",
            ("x2", "x3"): "state",
            ("x0", "x4"): "connected_to",
            ("y0", "y1"): "model",
            ("y0", "y2"): "country",
            ("y2", "y3"): "state",
            ("y0", "y4"): "connected_to",
            ("y3", "y5"): "city"
        },
        "node_types": {
            "x0": "Product", "x1": "", "x2": "", "x3": "", "x4": "",
            "y0": "Product", "y1": "", "y2": "", "y3": "", "y4": "", "y5": ""
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
            ("x0", "x1"), ("x0", "x2"), ("x2", "x3"), ("x3", "x4"), ("x0", "x5"), ("x0", "x6"), ("x0", "x7"),
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

plt.savefig("dual_graphs_with_lines.pdf", format="pdf")
plt.show()


