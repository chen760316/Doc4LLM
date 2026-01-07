"""
本版本已修正：对所有节点类型（不限于 Person）的 'name' 字段进行频率统计，
并对出现频率最高的 N 个核心节点进行 Wikidata 属性丰富。
"""

import json
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter

# --- 配置 ---
TOP_N_NODES = 50  # 选取前 50 个高频节点进行丰富
INPUT_FILE = "dense_property_graph.json"
OUTPUT_FILE = "dense_property_graph_enrich.json"

# --- Wikidata API 配置 ---
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def get_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    """配置带有重试机制的请求会话。"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        "User-Agent": "KnowledgeGraphEnricher/1.0 (contact@example.com) Python/3.x"
    })
    return session

def search_wikidata_entity(name: str) -> Optional[str]:
    """通过名称搜索 Wikidata 实体 QID。"""
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": name
    }
    try:
        session = get_retry_session()
        response = session.get(WIKIDATA_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = data.get("search", [])
        if results:
            return results[0]["id"]
    except requests.exceptions.RequestException as e:
        pass
    except ValueError:
        pass
    return None


def query_entity_properties(qid: str) -> Dict[str, str]:
    """查询给定 Wikidata 实体的基本属性。"""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    SELECT ?propLabel ?valLabel WHERE {{
      wd:{qid} ?p ?val .
      ?prop wikibase:directClaim ?p .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 50
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    properties = {}
    for result in results["results"]["bindings"]:
        prop = result["propLabel"]["value"]
        val = result["valLabel"]["value"]
        properties[prop] = properties.get(prop, val)
    return properties


# 🎯 关键修正：统计所有节点类型，并返回前 N 个名称
def get_top_n_nodes(graph_data: List[Dict[str, Any]], n: int) -> Set[str]:
    """
    统计所有节点的出现频率，并返回频率最高的前 n 个名称集合。
    """
    all_names = []
    for edge in graph_data:
        for node_key in ["node_1", "node_2"]:
            node = edge.get(node_key, {})
            # 统计所有节点的 name，不论 label 是什么
            if node.get("name"):
                all_names.append(node["name"])

    # 统计出现频率
    counts = Counter(all_names)

    # 获取出现频率最高的 N 个名称
    top_n_names = {name for name, count in counts.most_common(n)}

    print(f"[INFO] 发现 {len(counts)} 个不同的节点名称。")
    print(f"[INFO] 将丰富出现频率最高的 {len(top_n_names)} 个节点。")
    return top_n_names


# 全局缓存以存储已查询的 QID 和属性，避免重复查询和API调用
_WIKIDATA_CACHE: Dict[str, Optional[Dict[str, str]]] = {}
_QID_CACHE: Dict[str, Optional[str]] = {}

# 🎯 关键修正：enrich_node 现在接受 top_names，并对所有类型节点进行 QID 查找
def enrich_node(node: Dict[str, Any], top_names: Set[str]) -> Dict[str, Any]:
    """尝试 enrich 一个高频节点，并根据 label 决定添加哪些属性。"""

    name = node.get("name")
    if not name or name not in top_names:
        return node # 跳过非高频节点或无名节点

    node_label = node.get("label", "Unknown")

    # 检查是否已在缓存中
    qid = _QID_CACHE.get(name)
    if qid is not None and qid in _WIKIDATA_CACHE:
        # 缓存命中，无需 API 调用
        props = _WIKIDATA_CACHE[qid]
        if props:
            # 报告：已从缓存中获取
            # 统计已添加的属性数量
            added_count = 0

            # --- 仅执行第 3 步（属性合并） ---
            if node_label == "Person":
                enrich_fields = ["occupation", "country of citizenship", "educated at", "gender", "date of birth", "award received"]
            elif node_label == "Location":
                enrich_fields = ["country", "capital", "population", "coordinate location"]
            else:
                enrich_fields = ["instance of", "subclass of", "part of"]

            for field in enrich_fields:
                if field in props:
                    key = field.replace(" ", "_").lower()
                    if key not in node:
                        node[key] = props[field]
                        added_count += 1
            # ------------------------------------

            print(f"✔️ [DEBUG] 缓存命中: {name} (QID: {qid})。添加 {added_count} 个新属性。")
            return node
        else:
            # 报告：已缓存但无属性
            print(f"🚫 [DEBUG] 缓存命中: {name} (QID: {qid})，但无属性。")
            return node

    # 1. QID 查找 (缓存未命中或未查)
    if qid is None:
        qid = search_wikidata_entity(name)
        _QID_CACHE[name] = qid
        if not qid:
            print(f"❌ [DEBUG] 查找失败: {name}。未找到 QID。")
            # 缓存未找到 QID 的结果，避免重复查询
            _WIKIDATA_CACHE[qid] = None
            return node

    # 2. 属性查询 (缓存未命中)
    props = query_entity_properties(qid)
    _WIKIDATA_CACHE[qid] = props
    time.sleep(0.5) # 增加延迟，降低速率

    if not props:
        print(f"❌ [DEBUG] 查询失败: {name} (QID: {qid})。未找到可丰富属性。")
        return node

    # 3. 选取并合并属性
    added_count = 0
    if node_label == "Person":
        enrich_fields = ["occupation", "country of citizenship", "educated at", "gender", "date of birth", "award received"]
    elif node_label == "Location":
        enrich_fields = ["country", "capital", "population", "coordinate location"]
    else:
        enrich_fields = ["instance of", "subclass of", "part of"]

    for field in enrich_fields:
        if field in props:
            key = field.replace(" ", "_").lower()
            if key not in node:
                node[key] = props[field]
                added_count += 1

    print(f"✅ [DEBUG] 成功查询: {name} (QID: {qid})。添加 {added_count} 个新属性。")
    return node


def enrich_graph_nodes(data: List[Dict[str, Any]], top_names: Set[str]) -> List[Dict[str, Any]]:
    """遍历所有边中的节点进行属性丰富"""

    for item in data:
        # 对 node_1 进行丰富
        item["node_1"] = enrich_node(item["node_1"], top_names)
        # 对 node_2 进行丰富
        item["node_2"] = enrich_node(item["node_2"], top_names)
    return data


def main():
    # Step 1: 读取原始图谱 JSON 数据
    print(f"--- 🚀 正在加载文件: {INPUT_FILE} ---")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
    except Exception as e:
        print(f"[FATAL] 加载文件失败: {e}")
        return

    # Step 2: 筛选高频节点 (所有类型)
    top_names = get_top_n_nodes(graph_data, TOP_N_NODES)

    # Step 3: 节点丰富
    print("\n--- 🧠 开始节点丰富和调试报告 ---")
    enriched_data = enrich_graph_nodes(graph_data, top_names)

    # Step 4: 写入 enriched JSON
    print(f"\n--- ✅ 丰富完成，正在写入文件: {OUTPUT_FILE} ---")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    print(f"[DONE] 图谱已丰富并写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()