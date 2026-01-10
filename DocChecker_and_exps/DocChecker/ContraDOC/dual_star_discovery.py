"""
文档内提取双星结构，得到的双星模式中的每个星型结构为一跳
如果返回的重复实体出现次数正好为2，则其出现的两个三元组分别作为双星的两个星型，这个重复实体作为两个星型各自的中心，并且该重复实体用虚拟连接边连接
如果返回的重复实体出现次数大于等于3，则借助三元组中"edge_properties"作为聚类函数进行聚类，将所有该重复实体出现的三元组按照距离远近聚成两类，分别作为两个双星，然后该重复实体作为两个星星的中心，并按照虚拟连接边连接

此版本已禁用所有 PNG 图片生成。
"""
import json
import os
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import numpy as np
import copy
from typing import Dict, Any, List, Tuple, Set
import re # 导入 re 模块用于文件名清理

# 禁用 matplotlib 和 networkx 相关的导入，因为不再需要可视化
# import matplotlib
# import matplotlib.pyplot as plt
# import networkx as nx

# 假设 knowledge_graph_maker 和 dotenv 导入是外部依赖，这里保留
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LLM 客户端不再用于简化边标签，但仍需定义以避免知识图谱客户端依赖问题
llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# --- 文件名清理函数 ---

def clean_filename(name: str) -> str:
    """移除文件名中的非法字符和 Windows 不允许的保留字符"""
    cleaned = name.replace(' ', '_')
    # 移除所有非法和潜在问题的字符 (只保留字母、数字、下划线、连字符)
    # [^\w-] 匹配所有非单词字符和非连字符的字符
    cleaned = re.sub(r'[^\w-]', '', cleaned)
    return cleaned

# --- 核心逻辑函数 ---

def generate_virtual_links(star_uids: List[str]) -> List[Dict[str, Any]]:
    """
    生成虚拟边，连接所有星型结构的中心节点，表示同一实体
    """
    virtual_edges = []
    # 确保只连接唯一的实体
    unique_uids = list(set(star_uids))

    for i in range(len(unique_uids)):
        for j in range(i + 1, len(unique_uids)):
            # 虚拟连接边，使用原始实体名称
            virtual_edges.append({
                "node_1": {"name": unique_uids[i], "label": "VirtualCenter"},
                "node_2": {"name": unique_uids[j], "label": "VirtualCenter"},
                "relationship": "same_entity"
            })
    return virtual_edges

def extract_bipartite_stars(graph: List[Dict[str, Any]], central_entities: Dict[str, int]) -> List[Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]]:
    """
    提取双星结构：
    - 如果实体出现次数为2，直接作为双星结构
    - 如果实体出现次数>=3，使用KMeans进行聚类
    """
    stars = []

    for entity, count in central_entities.items():
        # 获取该实体作为 node_1 的所有三元组
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
            # 创建虚拟连接边 (针对原始实体)
            virtual_links = generate_virtual_links([entity])
            stars.append((star1, star2, virtual_links))
        elif count >= 3:
            # 如果实体出现次数>=3，使用KMeans进行聚类
            edge_properties = [entry['edge_properties'] for entry in triples]

            # 提取数值特征：使用 text_span 的质心
            X = []
            for prop in edge_properties:
                text_span = prop.get('text_span', {})
                start = text_span.get('start', 0)
                end = text_span.get('end', 0)

                # 使用 (start + end) / 2 作为特征
                centroid = (start + end) / 2
                X.append([centroid])

            X = np.array(X)

            # 使用KMeans聚类为2类
            # 增加对数据量的检查，防止 triples 数量小于 n_clusters
            if len(X) < 2:
                print(f"[WARN] 实体 {entity} 仅有 {len(X)} 个三元组，无法聚类为 2 类，跳过。")
                continue

            # n_init=10 是 KMeans 的推荐值
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)
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

# --- 禁用 LLM 简化和可视化函数 ---

def simplify_edge_labels_with_llm(relationships: List[str], llm: OpenAIClient) -> List[str]:
    """ 禁用 LLM 调用，直接返回原始关系 """
    return relationships


def visualize_property_graph_with_simplified_edges(json_data: List[Dict[str, Any]], llm: OpenAIClient, title: str="Knowledge Graph", output_file: str="graph.png") -> None:
    """ **此函数已被禁用以取消生成 PNG 文件** """
    # 由于不再生成图片，此函数留空
    print(f"[INFO] ⚠️ 跳过生成图片: {output_file}")
    pass


def save_and_visualize_dual_star(dual_star: List[Dict[str, Any]], virtual_links: List[Dict[str, Any]], output_dir: str) -> None:
    """
    保存双星结构为 JSON 文件，并跳过可视化双星结构
    """

    # 确保 dual_star 包含两个星型结构
    if len(dual_star) < 2:
        if dual_star:
            print(f"[WARN] 双星结构数据不足，跳过保存: {dual_star[0]['center']}")
        return

    # 🎯 关键修正：对中心实体名称使用 clean_filename
    center_name_raw = dual_star[0]['center']
    safe_uid = clean_filename(center_name_raw)

    # 修改 center 和 node_1.name 中的字段值，添加后缀 _1 和 _2
    def modify_center_and_node(center: str, triples: List[Dict[str, Any]], suffix: str) -> Tuple[str, List[Dict[str, Any]]]:
        """修改 center 和 node_1.name 中的字段值，给所有字段的值添加后缀（_1 或 _2）"""
        modified_center = center + "_" + suffix

        # 深度复制三元组，避免修改原数据
        modified_triples = copy.deepcopy(triples)

        for triple in modified_triples:
            if isinstance(triple.get('node_1'), dict) and 'name' in triple['node_1']:
                # 只修改中心实体（node_1）的 name，因为它是重复的
                if triple['node_1']['name'] == center:
                    triple['node_1']['name'] = triple['node_1']['name'] + "_" + suffix

        return modified_center, modified_triples

    # 修改两个 center 和 node_1.name 中的字段值
    star1, star2 = dual_star[0], dual_star[1]

    modified_star1_center, modified_star1_triples = modify_center_and_node(star1['center'], star1['triples'], '1')
    modified_star2_center, modified_star2_triples = modify_center_and_node(star2['center'], star2['triples'], '2')

    modified_star1 = {'center': modified_star1_center, 'triples': modified_star1_triples}
    modified_star2 = {'center': modified_star2_center, 'triples': modified_star2_triples}

    # JSON 文件路径
    json_filename = f"{output_dir}/dual_star_{safe_uid}.json"

    # 构造虚拟连接边 (使用带后缀的中心名称)
    modified_virtual_links = [{
        "node_1": {"name": modified_star1_center, "label": "VirtualCenter"},
        "node_2": {"name": modified_star2_center, "label": "VirtualCenter"},
        "relationship": "same_entity"
    }]

    # 将双星数据和虚拟连接边一起保存到一个字典
    full_dual_star = {
        'stars': [modified_star1, modified_star2],  # 保存双星结构
        'virtual_links': modified_virtual_links  # 保存虚拟连接边
    }

    # 保存为 JSON 文件
    with open(json_filename, "w", encoding='utf-8') as jf:
        json.dump(full_dual_star, jf, indent=2, ensure_ascii=False)
    print(f"✅ 保存了双星结构为 JSON: {json_filename}")

    # 跳过可视化步骤
    visualize_property_graph_with_simplified_edges([], llm, output_file=f"{output_dir}/dual_star_{safe_uid}.png")


# --- 辅助函数 ---

def count_node_1_entities(graph: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计作为 node_1 的实体出现次数 >= 2 的实体"""
    node_1_entities = [entry['node_1']['name'] for entry in graph if isinstance(entry.get('node_1'), dict) and 'name' in entry['node_1']]
    entity_counts = Counter(node_1_entities)
    central_entities = {entity: count for entity, count in entity_counts.items() if count >= 2}
    return central_entities

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """从文件中读取并解析JSON数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if not isinstance(data, list):
            print(f"[FATAL] JSON 文件内容不是列表。")
            return []
        return data
    except Exception as e:
        print(f"[FATAL] 加载文件失败 {file_path}: {e}")
        return []

# --- 主函数 ---

def main(input_file: str, output_dir: str):
    print(f"--- 🚀 开始提取双星结构 (禁用可视化) ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    merged_enriched_graph = load_json_data(input_file)
    if not merged_enriched_graph:
        return

    # 获取重复实体
    central_entities = count_node_1_entities(merged_enriched_graph)
    print(f"[INFO] 找到 {len(central_entities)} 个重复实体作为双星中心。")

    # 提取双星结构
    bipartite_stars_with_virtual_links = extract_bipartite_stars(merged_enriched_graph, central_entities)
    print(f"[INFO] 成功提取 {len(bipartite_stars_with_virtual_links)} 个双星结构。")

    # 打印并保存双星结构
    for idx, (star1, star2, virtual_links) in enumerate(bipartite_stars_with_virtual_links):
        print(f"\n--- 双星结构 {idx + 1}: {star1['center']} ---")
        # 仅保存 JSON，并跳过可视化
        save_and_visualize_dual_star([star1, star2], virtual_links, output_dir)


    print("\n--- ✅ 任务完成 ---")
    print(f"✨ 已成功生成所有双星结构 JSON 文件到 '{output_dir}' 目录。")


if __name__ == "__main__":
    # 输入文件路径和输出目录
    input_file = 'filtered_knowledge_graph_enrich.json'
    output_dir = './dual_star_output'

    main(input_file, output_dir)