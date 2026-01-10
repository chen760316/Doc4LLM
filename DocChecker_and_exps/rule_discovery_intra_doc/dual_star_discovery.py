"""
文档内提取双星结构，得到的双星模式中的每个星型结构为一跳
如果返回的重复实体出现次数正好为2，则其出现的两个三元组分别作为双星的两个星型，这个重复实体作为两个星型各自的中心，并且该重复实体用虚拟连接边连接
如果返回的重复实体出现次数大于等于3，则借助三元组中"edge_properties"作为聚类函数进行聚类，将所有该重复实体出现的三元组按照距离远近聚成两类，分别作为两个双星，然后该重复实体作为两个双星的中心，并按照虚拟连接边连接
"""
import json
import os
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
from knowledge_graph_maker import OpenAIClient
import copy
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

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
                "relationship": "same_entity"
            })
    return virtual_edges

def extract_bipartite_stars(graph, central_entities):
    """
    提取双星结构：
    - 如果实体出现次数为2，直接作为双星结构
    - 如果实体出现次数>=3，使用KMeans进行聚类
    """
    stars = []

    for entity, count in central_entities.items():
        # 获取该实体的所有三元组
        triples = [entry for entry in graph if entry['node_1']['name'] == entity]

        if count == 2:
            # 如果实体出现次数为2，直接构成双星结构
            star1 = {
                'center': entity,
                'triples': triples[:1],  # 第一组三元组
            }
            star2 = {
                'center': entity,
                'triples': triples[1:],  # 第二组三元组
            }
            # 创建虚拟连接边
            virtual_links = generate_virtual_links([entity])
            stars.append((star1, star2, virtual_links))
        elif count >= 3:
            # 如果实体出现次数>=3，使用KMeans进行聚类
            edge_properties = [entry['edge_properties'] for entry in triples]

            # 提取数值特征：假设edge_properties包含 'text_span' 字段
            X = []
            for prop in edge_properties:
                # 假设prop['text_span']包含 'start' 和 'end'
                text_span = prop.get('text_span', {})
                start = text_span.get('start', 0)  # 默认值为0
                end = text_span.get('end', 0)  # 默认值为0

                # 使用 (start + end) / 2 计算质心
                centroid = (start + end) / 2

                # 使用质心作为特征
                X.append([centroid])

            X = np.array(X)

            # 使用KMeans聚类为2类
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
            labels = kmeans.labels_

            star1 = {
                'center': entity,
                'triples': [triples[i] for i in range(len(triples)) if labels[i] == 0],  # 第一类三元组
            }
            star2 = {
                'center': entity,
                'triples': [triples[i] for i in range(len(triples)) if labels[i] == 1],  # 第二类三元组
            }
            # 创建虚拟连接边
            virtual_links = generate_virtual_links([entity])
            stars.append((star1, star2, virtual_links))

    return stars

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
    """
    Visualizes the property graph with simplified edge labels.
    """
    relationships = set()

    # Iterate over each star structure in the json_data
    for star in json_data:
        for entry in star['triples']:
            if "relationship" in entry:
                relationships.add(entry["relationship"])
            else:
                print(f"Warning: Missing 'relationship' key in entry: {entry}")

    # Convert to list for further processing
    relationships = list(relationships)

    if not relationships:
        print("Error: No valid relationships found in the input data.")
        return  # Exit the function if no relationships are found

    simplified_labels = simplify_edge_labels_with_llm(relationships, llm)
    rel_map = dict(zip(relationships, simplified_labels))

    G = nx.DiGraph()
    label_colors = {
        "Person": "#87CEFA", "Company": "#FFD700", "Event": "#FFB6C1",
        "Department": "#98FB98", "Location": "#DDA0DD", "default": "#D3D3D3"
    }

    # Iterate over each star and its triples to construct the graph
    for star in json_data:
        for entry in star['triples']:
            if "relationship" in entry:
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
    node_colors = [label_colors.get(G.nodes[n].get("label", "default"), "#D3D3D3") for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, edgecolors="black", ax=ax)

    node_labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, ax=ax)

    non_same_entity_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("orig_rel") != "same_entity"]
    nx.draw_networkx_edges(G, pos, edgelist=non_same_entity_edges, arrowstyle='-|>', arrowsize=40, edge_color='gray', ax=ax)

    same_entity_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("orig_rel") == "same_entity"]
    nx.draw_networkx_edges(G, pos, edgelist=same_entity_edges, arrowstyle='<->', arrowsize=45, edge_color='red', width=2, ax=ax)

    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', ax=ax)

    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[SAVED] 图已保存为: {output_file}")


def save_and_visualize_dual_star(dual_star, virtual_links, output_dir):
    """
    保存双星结构为 JSON 文件，并可视化双星结构
    :param dual_star: 双星结构数据
    :param virtual_links: 虚拟连接边
    :param output_dir: 输出目录
    """
    # 创建一个安全的文件名，去掉不合法字符
    safe_uid1 = dual_star[0]['center'].replace(" ", "_").replace("/", "_")
    safe_uid2 = dual_star[1]['center'].replace(" ", "_").replace("/", "_")

    # 修改 center 和 node_1.name 中的字段值，添加后缀 _1 和 _2
    def modify_center_and_node(center, triples, suffix):
        """
        修改 center 和 node_1.name 中的字段值，给所有字段的值添加后缀（_1 或 _2）
        """
        # 如果center是字符串，将其值后加后缀
        if isinstance(center, str):
            center = center + "_" + suffix

        # 修改每个三元组中的node_1.name
        for triple in triples:
            if isinstance(triple['node_1'], dict) and 'name' in triple['node_1']:
                triple['node_1']['name'] = triple['node_1']['name'] + "_" + suffix

        return center, triples

    # 修改两个 center 和 node_1.name 中的字段值
    modified_star1 = copy.deepcopy(dual_star[0])
    modified_star2 = copy.deepcopy(dual_star[1])

    modified_star1['center'], modified_star1['triples'] = modify_center_and_node(modified_star1['center'], modified_star1['triples'], '1')
    modified_star2['center'], modified_star2['triples'] = modify_center_and_node(modified_star2['center'], modified_star2['triples'], '2')

    # JSON 文件路径
    json_filename = f"{output_dir}/dual_star_{safe_uid1}_vs_{safe_uid2}.json"

    # 将双星数据和虚拟连接边一起保存到一个字典
    full_dual_star = {
        'stars': [modified_star1, modified_star2],  # 保存双星结构
        'virtual_links': virtual_links  # 保存虚拟连接边
    }

    # 合并 star1、star2 和虚拟边
    combined = modified_star1['triples'] + modified_star2['triples'] + [{
        "node_1": {"name": safe_uid1, "label": "VirtualCenter"},
        "node_2": {"name": safe_uid2, "label": "VirtualCenter"},
        "relationship": "same_entity"
    }]

    # 保存为 JSON 文件
    with open(json_filename, "w", encoding='utf-8') as jf:
        json.dump(full_dual_star, jf, indent=2, ensure_ascii=False)
    print(f"保存了双星结构为 JSON: {json_filename}")

    # 可视化双星结构（包括虚拟连接边）
    visualize_property_graph_with_simplified_edges(
        full_dual_star['stars'] + virtual_links,  # 传入双星结构和虚拟连接边
        llm,  # 生成图所需的 LLM
        title=f"Dual-Star Structure: {safe_uid1} ↔ {safe_uid2}",  # 图的标题
        output_file=f"{output_dir}/dual_star_{safe_uid1}_vs_{safe_uid2}.png"  # 输出图像文件名
    )

    print(f"双星结构图像已保存为 PNG: {output_dir}/dual_star_{safe_uid1}_vs_{safe_uid2}.png")




def count_node_1_entities(graph):
    # 提取所有node_1的实体名称
    node_1_entities = [entry['node_1']['name'] for entry in graph]

    # 使用Counter统计每个实体出现的次数
    entity_counts = Counter(node_1_entities)

    # 过滤出出现次数大于等于2的实体
    central_entities = {entity: count for entity, count in entity_counts.items() if count >= 2}

    return central_entities




def load_json_data(file_path):
    # 从文件中读取并解析JSON数据
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def main(input_file, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 从文件加载数据
    merged_enriched_graph = load_json_data(input_file)

    # 获取重复实体
    central_entities = count_node_1_entities(merged_enriched_graph)

    # 初始化变量
    star_uid_mapping = []  # 用于存储所有虚拟中心的唯一ID
    all_flat_triples = []  # 用于存储所有的三元组（包括虚拟边）

    # 提取双星结构
    bipartite_stars_with_virtual_links = extract_bipartite_stars(merged_enriched_graph, central_entities)

    # 打印并保存双星结构
    for idx, (star1, star2, virtual_links) in enumerate(bipartite_stars_with_virtual_links):
        print(f"双星结构 {idx + 1}:")
        print(f"  星型 1: {star1}")
        print(f"  星型 2: {star2}")

        # 保存并可视化双星结构
        save_and_visualize_dual_star([star1, star2], virtual_links, output_dir)

        # 更新所有三元组和虚拟中心ID映射
        all_flat_triples.extend([star1, star2])  # 这里假设 star1 和 star2 已经是扁平化的三元组
        star_uid_mapping.extend([star1["center"], star2["center"]])  # 你可能需要从星型结构中提取中心节点

    # Step 6: 去重UID后，连接所有虚拟中心，构建“same_entity”的总图
    unique_star_uids = list(set(star_uid_mapping))  # 获取唯一的中心节点ID
    virtual_links = generate_virtual_links(unique_star_uids)  # 生成虚拟边
    all_flat_triples.extend(virtual_links)  # 将虚拟边添加到所有三元组中

    # 可视化并保存最终图
    save_and_visualize_dual_star(all_flat_triples, [], output_dir)





if __name__ == "__main__":
    # 输入文件路径和输出目录
    input_file = 'merged_enriched_graph.json'  # 请根据实际文件路径修改
    output_dir = './dual_star_output'  # 输出目录

    # 调用main函数
    main(input_file, output_dir)