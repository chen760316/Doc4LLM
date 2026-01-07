"""
--- ✅ 写入文件: filtered_knowledge_graph_enrich.json ---
[DONE] 处理完成
总耗时: 31.31 秒
"""
import json
import asyncio
import aiohttp
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter

# --- 配置 ---
TOP_N_NODES = 50
CONCURRENT_REQUESTS = 5  # 控制并发数，避免被 Wikidata 封锁
INPUT_FILE = "filtered_knowledge_graph.json"
OUTPUT_FILE = "filtered_knowledge_graph_enrich.json"

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# 全局缓存
_WIKIDATA_CACHE: Dict[str, Optional[Dict[str, str]]] = {}
_QID_CACHE: Dict[str, Optional[str]] = {}

# -------------------- 1. 核心工具函数 --------------------

async def fetch_json(session: aiohttp.ClientSession, url: str, params: Dict = None) -> Optional[Dict]:
    """通用的异步获取 JSON 函数"""
    try:
        async with session.get(url, params=params, timeout=10) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429: # Too Many Requests
                print("⚠️ [WARNING] 触发频率限制，等待 5 秒...")
                await asyncio.sleep(5)
    except Exception as e:
        pass
    return None

async def search_wikidata_entity(session: aiohttp.ClientSession, name: str) -> Optional[str]:
    """异步搜索 QID"""
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": name
    }
    data = await fetch_json(session, WIKIDATA_SEARCH_URL, params)
    if data and data.get("search"):
        return data["search"][0]["id"]
    return None

async def query_entity_properties(session: aiohttp.ClientSession, qid: str) -> Dict[str, str]:
    """异步执行 SPARQL 查询"""
    query = f"""
    SELECT ?propLabel ?valLabel WHERE {{
      wd:{qid} ?p ?val .
      ?prop wikibase:directClaim ?p .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 50
    """
    params = {"query": query, "format": "json"}
    data = await fetch_json(session, SPARQL_ENDPOINT, params)

    properties = {}
    if data:
        for result in data["results"]["bindings"]:
            prop = result["propLabel"]["value"]
            val = result["valLabel"]["value"]
            properties[prop] = properties.get(prop, val)
    return properties

# -------------------- 2. 节点丰富逻辑 --------------------

async def enrich_single_node(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                             node: Dict[str, Any], top_names: Set[str]):
    """异步丰富单个节点，使用信号量控制并发"""
    name = node.get("name")
    if not name or name not in top_names:
        return

    # 1. 检查 QID 和属性缓存
    if name in _QID_CACHE:
        qid = _QID_CACHE[name]
    else:
        async with semaphore: # 进入限流区域
            qid = await search_wikidata_entity(session, name)
            _QID_CACHE[name] = qid
            await asyncio.sleep(0.2) # 微小延迟

    if not qid:
        return

    if qid in _WIKIDATA_CACHE:
        props = _WIKIDATA_CACHE[qid]
    else:
        async with semaphore:
            props = await query_entity_properties(session, qid)
            _WIKIDATA_CACHE[qid] = props
            await asyncio.sleep(0.3)

    if not props:
        return

    # 2. 合并属性 (逻辑与同步版一致)
    node_label = node.get("label", "Unknown")
    if node_label == "Person":
        fields = ["occupation", "country of citizenship", "educated at", "gender", "date of birth", "award received"]
    elif node_label == "Location":
        fields = ["country", "capital", "population", "coordinate location"]
    else:
        fields = ["instance of", "subclass of", "part of"]

    added_count = 0
    for field in fields:
        if field in props:
            key = field.replace(" ", "_").lower()
            if key not in node:
                node[key] = props[field]
                added_count += 1

    if added_count > 0:
        print(f"✅ [DEBUG] 异步丰富完成: {name} ({qid})，添加 {added_count} 属性")

# -------------------- 3. 主程序 --------------------

def get_top_n_nodes(graph_data: List[Dict[str, Any]], n: int) -> Set[str]:
    """统计逻辑保持不变"""
    all_names = [edge[k]["name"] for edge in graph_data for k in ["node_1", "node_2"] if edge.get(k, {}).get("name")]
    return {name for name, count in Counter(all_names).most_common(n)}

async def main():
    print(f"--- 🚀 正在加载文件: {INPUT_FILE} ---")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
    except Exception as e:
        print(f"[FATAL] 加载文件失败: {e}"); return

    top_names = get_top_n_nodes(graph_data, TOP_N_NODES)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    headers = {"User-Agent": "KnowledgeGraphEnricherAsync/1.0 (contact@example.com)"}

    async with aiohttp.ClientSession(headers=headers) as session:
        print(f"--- 🧠 开始异步节点丰富 (最大并发: {CONCURRENT_REQUESTS}) ---")

        # 收集所有需要丰富的节点任务
        tasks = []
        for edge in graph_data:
            tasks.append(enrich_single_node(session, semaphore, edge["node_1"], top_names))
            tasks.append(enrich_single_node(session, semaphore, edge["node_2"], top_names))

        # 并发执行
        await asyncio.gather(*tasks)

    print(f"\n--- ✅ 写入文件: {OUTPUT_FILE} ---")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"[DONE] 处理完成")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    print(f"总耗时: {time.time() - start_time:.2f} 秒")