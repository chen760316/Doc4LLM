"""
========================================
✅ 任务全部完成！
✨ 输出目录: 'dual_star_json'
⏱️ 总程序运行时间: 8.23 秒
========================================
"""
import json
import os
import copy
import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Set

# 导入 KMeans 和 numpy
from sklearn.cluster import KMeans
import numpy as np
import time

# 假设 knowledge_graph_maker 和 dotenv 导入是外部依赖
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import hashlib # 需要在文件顶部导入

# 加载环境变量
load_dotenv()
# LLM 客户端定义 (虽然未被使用，但保留了环境初始化)
# 🚨 注意：这里会打印 API Key 相关的警告，但代码功能不受影响。
try:
    llm = OpenAIClient(
        model="gpt-5.2",
        temperature=0.1,
        top_p=0.9,
        max_tokens=1500
    )
except Exception:
    # 捕获并忽略 API Key 缺失的初始化错误，因为后续代码不依赖 LLM
    pass


# --- 辅助函数：文件加载与清理 ---

def load_graph(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSON 文件，并明确指定 UTF-8 编码以避免 UnicodeDecodeError"""
    try:
        # 明确指定 encoding='utf-8' 来处理包含中文和特殊字符的JSON文件
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[ERROR] JSON file content is not a list in {file_path}")
            return []
        return data
    except Exception as e:
        print(f"[FATAL] Failed to load JSON file {file_path}: {e}")
        return []

def clean_filename(name: str) -> str:
    """移除文件名中的非法字符"""
    cleaned = name.replace(' ', '_')
    # 移除所有非法和潜在问题的字符 (只保留字母、数字、下划线、连字符)
    cleaned = re.sub(r'[^\w-]', '', cleaned)
    return cleaned

def count_node_1_entities(graph: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计作为 node_1 的实体出现次数 >= 2 的实体"""
    # 🎯 安全访问 'node_1' 和 'name'
    node_1_entities = [
        entry['node_1']['name'] for entry in graph
        if isinstance(entry.get('node_1'), dict) and 'name' in entry['node_1']
    ]
    entity_counts = Counter(node_1_entities)
    # 过滤出出现次数大于等于2的实体
    central_entities = {entity: count for entity, count in entity_counts.items() if count >= 2}
    return central_entities

# --- 核心逻辑函数：多跳星型结构生成 ---

def build_star_structure(center_node: str, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    为给定的实体和其相关三元组创建初始的单跳星型结构。
    """
    neighbors = []
    edges = []

    for triple in triples:
        node_2 = triple.get("node_2", {})
        node_2_name = node_2.get("name")
        relationship = triple.get("relationship")

        if node_2_name and relationship:
            neighbors.append(node_2_name)
            edges.append(relationship)
        else:
            print(f"[WARN] 忽略三元组：中心 '{center_node}' 的邻居节点或关系缺失: {triple}")

    return {
        "center": center_node,
        "neighbors": neighbors,
        "edges": edges,
        "multi_hop_neighbors": []  # 初始化为空
    }


def get_neighbors_with_hops(node: str, hop_count: int, max_hops: int, all_triples: List[Dict[str, Any]], visited_triples: Set[Tuple] = set(), multi_hop_data: Set[Tuple] = set()) -> Set[Tuple]:
    """
    递归获取指定节点的多跳邻居。
    """
    if hop_count > max_hops:
        return multi_hop_data

    # 找到以当前节点作为 node_1 的三元组
    current_triples = [
        triple for triple in all_triples
        if isinstance(triple.get("node_1"), dict) and
           triple["node_1"].get("name") == node
    ]

    newly_visited_triples = set()
    next_hop_nodes = []

    for triple in current_triples:

        # 关键修正：安全地提取所有必要字段，并跳过不完整的三元组
        node_1_name = triple.get("node_1", {}).get("name")
        node_2_name = triple.get("node_2", {}).get("name")
        relationship = triple.get("relationship")

        # 如果任何关键字段缺失，则跳过这个三元组
        if not node_1_name or not node_2_name or not relationship:
            continue

        # 使用不可变的值创建可哈希的唯一标识符
        doc_id = triple.get("edge_properties", {}).get("doc_id", "")
        text_span = triple.get("edge_properties", {}).get("text_span", {})
        span_start = text_span.get("start", -1)
        span_end = text_span.get("end", -1)

        triple_id = (node_1_name, node_2_name, relationship, doc_id, span_start, span_end)

        # 避免重复访问三元组
        if triple_id in visited_triples:
            continue

        newly_visited_triples.add(triple_id)

        neighbor = node_2_name

        # 记录多跳关系 (邻居, 跳数, 关系)
        multi_hop_data.add((neighbor, hop_count, relationship))

        if hop_count < max_hops:
            next_hop_nodes.append(neighbor)

    visited_triples.update(newly_visited_triples)

    # 递归查找下一跳
    for next_node in next_hop_nodes:
        # 递归调用时，更新 visited_triples 和 multi_hop_data
        multi_hop_data = get_neighbors_with_hops(
            next_node,
            hop_count + 1,
            max_hops,
            all_triples,
            visited_triples,
            multi_hop_data
        )

    return multi_hop_data

def remove_single_hop_neighbors(star: Dict[str, Any]) -> Dict[str, Any]:
    """
    删除多跳邻居中的跳数为1的邻居，
    并删除与 neighbors (单跳) 中节点重复的 multi_hop_neighbors 节点
    """
    neighbor_nodes = set(star["neighbors"])
    unique_multi_hop_neighbors = []
    seen_multi_hop_nodes = set()

    for mh_neighbor in star["multi_hop_neighbors"]:
        neighbor = mh_neighbor["neighbor"]
        hop_count = mh_neighbor["hop_count"]

        # 1. 删除跳数为 1 的邻居（单跳关系已经在 'neighbors' 中）
        if hop_count <= 1:
            continue

        # 2. 删除与单跳邻居重复的节点
        if neighbor in neighbor_nodes:
            continue

        # 3. 确保多跳邻居列表自身没有重复项
        if neighbor not in seen_multi_hop_nodes:
            seen_multi_hop_nodes.add(neighbor)
            unique_multi_hop_neighbors.append(mh_neighbor)

    star["multi_hop_neighbors"] = unique_multi_hop_neighbors
    return star


def merge_star_structures_with_hops(initial_stars: List[Dict[str, Any]], all_triples: List[Dict[str, Any]], max_hops: int=5) -> List[Dict[str, Any]]:
    """ 为每个初始星型结构添加多跳邻居 """
    merged_stars = []

    for star in initial_stars:
        center = star["center"]

        # 获取该中心节点的完整多跳关系 (neighbor, hop_count, edge)
        multi_hop_data = get_neighbors_with_hops(center, 1, max_hops, all_triples, set(), set())

        # 将多跳邻居添加到星型结构中
        star["multi_hop_neighbors"] = [{
            "neighbor": neighbor,
            "hop_count": hop_count,
            "edge": edge
        } for neighbor, hop_count, edge in multi_hop_data]

        # 删除单跳邻居，并确保多跳邻居中没有重复项
        star = remove_single_hop_neighbors(star)

        merged_stars.append(star)

    return merged_stars

# --- 双星结构生成与保存 ---

def save_dual_star_structure(dual_star: List[Dict[str, Any]], center_name: str, output_dir: str):
    """保存多跳双星结构到 JSON 文件夹，解决长路径和非法字符问题"""

    # 1. 基础清理
    safe_center_name = clean_filename(center_name)

    # 2. 解决 Windows 路径过长问题 (MAX_PATH 限制)
    # 如果处理后的文件名超过 50 个字符，使用哈希值代替部分文件名
    if len(safe_center_name) > 50:
        name_hash = hashlib.md5(center_name.encode('utf-8')).hexdigest()[:8]
        safe_center_name = f"{safe_center_name[:40]}_{name_hash}"

    file_name = os.path.join(output_dir, f"dual_star_{safe_center_name}_mh.json")

    # 确保目录存在 (双重保险)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if len(dual_star) < 2:
        print(f"[ERROR] 无法为 {center_name} 创建双星结构。")
        return

    star1_center = dual_star[0]['center']
    star2_center = dual_star[1]['center']

    # 添加虚拟连接边
    virtual_links = [{
        "node_1": {"name": star1_center, "label": "VirtualCenter"},
        "node_2": {"name": star2_center, "label": "VirtualCenter"},
        "relationship": "same_entity"
    }]

    output_data = {
        'entity_full_name': center_name,  # 在文件内部保留完整名称
        'stars': dual_star,
        'virtual_links': virtual_links
    }

    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"✅ 双星 JSON 文件已保存：{file_name}")
    except Exception as e:
        print(f"[FATAL] 写入文件失败 {file_name}: {e}")

def extract_bipartite_stars_multi_hop(graph: List[Dict[str, Any]], central_entities: Dict[str, int], max_hops: int):
    """
    多跳双星结构提取的主函数。
    """
    output_dir = "dual_star_json"
    os.makedirs(output_dir, exist_ok=True)

    for entity, count in central_entities.items():
        # 🎯 修正点：在这里也进行安全访问，防止 KeyError: 'name'
        all_entity_triples = [
            entry for entry in graph
            if isinstance(entry.get('node_1'), dict) and
               entry['node_1'].get('name') == entity
        ]

        if not all_entity_triples:
            print(f"[WARN] 实体 {entity} 在当前图数据中没有有效的三元组，跳过。")
            continue

        triples_A = []
        triples_B = []

        if count == 2:
            # count=2：直接拆分
            triples_A = all_entity_triples[:1]
            triples_B = all_entity_triples[1:]
            print(f"\n[INFO] 实体 {entity} (count={count}): 直接拆分为两组三元组。")

        elif count >= 3:
            # count>=3：KMeans聚类

            # 确保三元组有 'edge_properties' 和 'text_span'
            valid_triples_for_clustering = [
                entry for entry in all_entity_triples
                if isinstance(entry.get('edge_properties'), dict) and
                   isinstance(entry['edge_properties'].get('text_span'), dict)
            ]

            if len(valid_triples_for_clustering) < 2:
                print(f"\n[WARN] 实体 {entity} 仅有 {len(valid_triples_for_clustering)} 个有效三元组，无法聚类为 2 类，跳过。")
                continue

            # 重新映射三元组和属性，只使用有效的
            all_entity_triples = valid_triples_for_clustering

            edge_properties = [entry['edge_properties'] for entry in all_entity_triples]

            # 提取 text_span 的质心作为特征
            X = []
            for prop in edge_properties:
                text_span = prop.get('text_span', {})
                start = text_span.get('start', 0)
                end = text_span.get('end', 0)
                centroid = (start + end) / 2
                X.append([centroid])

            X = np.array(X)

            # 重新检查长度 (理论上 >= 2)
            if len(X) < 2:
                continue

            # 抑制 KMeans 的警告
            # FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(X)

            labels = kmeans.labels_

            triples_A = [all_entity_triples[i] for i in range(len(all_entity_triples)) if labels[i] == 0]
            triples_B = [all_entity_triples[i] for i in range(len(all_entity_triples)) if labels[i] == 1]
            print(f"\n[INFO] 实体 {entity} (count={count}): 使用 KMeans 划分为 A({len(triples_A)}) 和 B({len(triples_B)}) 组三元组。")

        if not triples_A or not triples_B:
            print(f"[WARN] 实体 {entity} 划分后存在空集，跳过。")
            continue

        # -----------------------------------------------
        # Step 2: 对 Triples A 和 Triples B 各自生成多跳星型结构
        # -----------------------------------------------

        # 1. Star A 生成
        initial_stars_A = [build_star_structure(entity, triples_A)]
        star_A = merge_star_structures_with_hops(initial_stars_A, graph, max_hops=max_hops)

        # 2. Star B 生成
        initial_stars_B = [build_star_structure(entity, triples_B)]
        star_B = merge_star_structures_with_hops(initial_stars_B, graph, max_hops=max_hops)

        # 确保成功生成
        if star_A and star_B:
            # Step 3: 合并 Star A 和 Star B 并保存
            dual_star = [star_A[0], star_B[0]] # 仅取第一个（且唯一的）星型结构
            save_dual_star_structure(dual_star, entity, output_dir)
        else:
            print(f"[WARN] 未能为实体 {entity} 成功生成双星结构，跳过保存。")


def main():
    # 记录程序开始的绝对时间
    overall_start_time = time.time()

    input_file = "filtered_dense_property_graph_enrich.json"
    max_hops = 3
    print(f"--- 🚀 开始提取多跳双星结构 (Max Hops={max_hops}) ---")

    # 确保输出目录存在
    os.makedirs("dual_star_json", exist_ok=True)

    # --- Step 1: 加载阶段 ---
    load_start = time.time()
    full_graph_triples = load_graph(input_file)
    if not full_graph_triples:
        return
    print(f"[TIMER] 数据加载完成，耗时: {time.time() - load_start:.2f}s")

    # --- Step 2: 统计阶段 ---
    count_start = time.time()
    central_entities = count_node_1_entities(full_graph_triples)
    print(f"[INFO] 找到 {len(central_entities)} 个重复实体作为双星中心。")
    print(f"[TIMER] 实体统计耗时: {time.time() - count_start:.4f}s")

    # --- Step 3: 核心处理与保存阶段 ---
    processing_start = time.time()
    extract_bipartite_stars_multi_hop(full_graph_triples, central_entities, max_hops)
    print(f"[TIMER] 核心逻辑处理（聚类+多跳搜索+保存）总耗时: {time.time() - processing_start:.2f}s")

    # --- 总结 ---
    total_elapsed = time.time() - overall_start_time
    print("\n" + "=" * 40)
    print(f"✅ 任务全部完成！")
    print(f"✨ 输出目录: 'dual_star_json'")
    print(f"⏱️ 总程序运行时间: {total_elapsed:.2f} 秒")
    print("=" * 40)


if __name__ == "__main__":
    main()