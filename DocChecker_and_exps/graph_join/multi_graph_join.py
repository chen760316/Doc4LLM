"""
图连接后得到merged_graph.json和merged_graph.png两个文件
useful
"""
import json
import copy
import networkx as nx
import matplotlib

# 确保在非GUI环境下也能保存图片
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt


# ----------------- 核心函数保持不变 -----------------

def build_graph(json_data, prefix=""):
    G = nx.DiGraph()
    # name_map 存储 {原实体名: 前缀后的节点ID}
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

        # 使用前缀确保不同图中的同名实体有唯一的节点ID
        prefixed_name1 = f"{prefix}:{name1}"
        prefixed_name2 = f"{prefix}:{name2}"

        # 记录原始名称到新的ID映射
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


def add_cross_graph_connections(G, graph_maps):
    """
    添加所有子图之间的跨图连接（epsilon-edges）。

    G: 组合后的总图
    graph_maps: 包含所有子图 name_map 的列表: [map1, map2, map3, ...]
    """
    # 遍历所有可能的两两组合
    for i in range(len(graph_maps)):
        for j in range(i + 1, len(graph_maps)):
            map1 = graph_maps[i]
            map2 = graph_maps[j]

            # 查找 map1 和 map2 之间共指的实体
            for original_name in map1:
                if original_name in map2:
                    node1 = map1[original_name]
                    node2 = map2[original_name]

                    # 仅在边不存在时添加，防止重复
                    if not G.has_edge(node1, node2) and not G.has_edge(node2, node1):
                        # 添加 epsilon-edge (这里使用 label="Connected")
                        G.add_edge(node1, node2,
                                   label="epsilon",  # 使用epsilon作为标签
                                   style="dashed",
                                   color="blue",
                                   type="coreference")
                        # 为了可视化效果，可以添加反向边，但这不是必须的
                        # G.add_edge(node2, node1, label="epsilon", style="dashed", color="blue", type="coreference")


def visualize_combined_graph(G):
    # 可视化逻辑保持不变，确保能处理多图
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
        # 排除用于绘图的 style, color
        other_props = {k: v for k, v in d.items() if k not in ['label', 'style', 'color']}
        if other_props:
            props_str = ", ".join(f"{k}: {v}" for k, v in other_props.items())
            label = f"{label}\n[{props_str}]"

        # 对于 epsilon 边，仅显示 label
        if d.get('label') == 'epsilon':
            label = '$\epsilon$'

        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', ax=ax)

    # 右侧显示所有节点属性文本
    attr_texts = []
    for node in G.nodes():
        attrs = G.nodes[node].copy()
        label = attrs.pop("label", None)
        name = attrs.pop("original_name", node)
        # 排除在节点属性中可能残留的临时键
        attrs = {k: v for k, v in attrs.items() if k not in ['id', 'original_name']}
        if attrs:
            text = f"{name} ({node.split(':')[0]}) Attributes:\n" + "\n".join(f"  {k}: {v}" for k, v in attrs.items())
            attr_texts.append(text)

    full_attr_text = "\n\n".join(attr_texts)

    plt.subplots_adjust(right=0.7)
    ax.text(1.02, 0.5, full_attr_text, fontsize=10, va='center', ha='left', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))

    plt.title("Connected Knowledge Graphs with Full Attributes (Multi-Graph Fusion)", fontsize=18)
    plt.axis("off")
    plt.savefig("merged_graph.png", dpi=300)
    # plt.show() # 如果在无GUI环境运行，应注释此行


# ----------------- MAIN 函数修改 -----------------

def main():
    # 假设输入文件列表
    input_files = ["graph1.json", "graph2.json", "graph3.json"]  # 可以根据需要增加文件

    all_graphs = []
    all_maps = []
    G_combined = nx.DiGraph()  # 初始化组合图

    print(f"Reading and building graphs from {input_files}...")

    # 循环处理所有输入文件
    for i, file_name in enumerate(input_files):
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            prefix = f"G{i + 1}"
            G_i, map_i = build_graph(json_data, prefix=prefix)

            # 使用 compose 逐步合并图，实现 disjoint union
            G_combined = nx.compose(G_combined, G_i)
            all_maps.append(map_i)
            print(f"Successfully built and added {file_name} (Prefix: {prefix}).")

        except FileNotFoundError:
            print(f"Warning: File not found: {file_name}. Skipping.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_name}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred processing {file_name}: {e}. Skipping.")

    if not G_combined.nodes:
        print("No graphs were successfully loaded and combined. Exiting.")
        return

    # 添加所有跨图连接边
    print("Adding cross-graph connections (epsilon-edges)...")
    add_cross_graph_connections(G_combined, all_maps)

    # 可视化
    print("Visualizing and saving graph to merged_graph.png...")
    visualize_combined_graph(G_combined)

    # 导出为 JSON
    merged_triples = []

    for u, v, data in G_combined.edges(data=True):
        node1 = G_combined.nodes[u]
        node2 = G_combined.nodes[v]

        # 构造节点数据，保留所有属性
        node1_data = {
            "name": node1.get("original_name", u),
            **{k: v for k, v in node1.items() if k not in ["original_name", "id"]}
        }

        node2_data = {
            "name": node2.get("original_name", v),
            **{k: v for k, v in node2.items() if k not in ["original_name", "id"]}
        }

        # 将边的非标签属性完整加入 edge_properties
        edge_props = {k: v for k, v in data.items() if k != "label"}

        triple = {
            "node_1": node1_data,
            "node_2": node2_data,
            "relationship": data.get("label", "related_to")
        }

        if edge_props:
            triple["edge_properties"] = edge_props

        merged_triples.append(triple)

    with open("merged_graph.json", "w", encoding="utf-8") as out_file:
        json.dump(merged_triples, out_file, ensure_ascii=False, indent=4)

    print("---")
    print("Fusion complete.")
    print(f"Total nodes in merged graph: {len(G_combined.nodes)}")
    print(f"Total edges in merged graph: {len(G_combined.edges)}")
    print("Merged graph JSON written to merged_graph.json")
    print("Merged graph image written to merged_graph.png")


if __name__ == "__main__":
    main()