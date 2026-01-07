"""
useful
"""
import json
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import requests

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def search_wikidata_entity(name):
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": name
    }
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        })
        response = session.get(WIKIDATA_SEARCH_URL, params=params, timeout=5)
        try:
            response = session.get(WIKIDATA_SEARCH_URL, params=params, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 请求失败: {e}")
            return None  # 或其他处理逻辑
        # session = requests.Session()
        #
        # headers = {
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        # }
        #
        # response = session.get(WIKIDATA_SEARCH_URL, params=params, headers=headers, timeout=5)
        # response.raise_for_status()  # 如果状态码不是200，会抛异常

        # 尝试解析json，捕获异常
        try:
            data = response.json()
        except ValueError:
            print(f"[ERROR] 响应不是有效JSON: {response.text[:200]}")  # 打印部分内容看下

            return None

        results = data.get("search", [])
        if results:
            return results[0]["id"]
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 请求失败: {e}")
    return None


def query_entity_properties(qid):
    """查询给定 Wikidata 实体的基本属性"""
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
        if prop not in properties:
            properties[prop] = val
    return properties


def enrich_node(node):
    """尝试 enrich 一个 Person 节点"""
    if node.get("label") != "Person":
        return node

    name = node.get("name")
    qid = search_wikidata_entity(name)
    if not qid:
        print(f"[WARN] 没有找到 Wikidata 实体: {name}")
        return node

    print(f"[INFO] 正在丰富节点: {name} (QID: {qid})")
    try:
        props = query_entity_properties(qid)
        # 选取一些常见的字段进行 enrich
        enrich_fields = ["occupation", "country of citizenship", "educated at", "gender", "award received"]
        for field in enrich_fields:
            if field in props:
                node[field.replace(" ", "_").lower()] = props[field]
    except Exception as e:
        print(f"[ERROR] SPARQL 查询失败: {name} - {e}")

    time.sleep(1)  # 避免对 Wikidata 查询过快
    return node


def enrich_graph_nodes(data):
    """遍历所有节点进行属性丰富"""
    for item in data:
        item["node_1"] = enrich_node(item["node_1"])
        item["node_2"] = enrich_node(item["node_2"])
    return data


def main():
    # Step 1: 读取原始图谱 JSON 数据
    input_file = "graph1.json"
    output_file = "graph1_enriched.json"
    with open(input_file, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # Step 2: 节点丰富
    enriched_data = enrich_graph_nodes(graph_data)

    # Step 3: 写入 enriched JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)

    print(f"[DONE] 图谱已丰富并写入: {output_file}")


if __name__ == "__main__":
    main()
