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

def load_json_from_directory(directory):
    """读取目录中的所有 JSON 文件并返回内容"""
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                json_data.append(data)
    return json_data

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
    """可视化双星图，并简化关系标签"""
    import networkx as nx
    import matplotlib.pyplot as plt
    import copy

    # 从 json_data 中提取 "triples" 键
    triples = json_data.get("triples", [])
    if not triples:
        print("No triples found in the data!")
        return

    # 提取所有的关系字段
    relationships = list({entry["relationship"] for entry in triples})
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

    for entry in triples:
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

    # 生成布局
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

def main():
    # 设置目录路径，确保读取 JSON 文件
    input_directory = "./dual_star_output"  # 修改为你的 JSON 文件目录路径

    # 读取目录中的 JSON 文件
    json_data_list = load_json_from_directory(input_directory)

    # 遍历每个 JSON 文件并可视化
    for i, json_data in enumerate(json_data_list):
        title = f"Dual-Star Structure {i+1}"
        output_file = f"./dual_star_output/dual_star_{i+1}.png"
        visualize_property_graph_with_simplified_edges(json_data, llm, title, output_file)

if __name__ == "__main__":
    main()
