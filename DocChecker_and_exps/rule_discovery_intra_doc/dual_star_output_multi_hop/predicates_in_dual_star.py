"""
text1.txt和text2.txt不能和inter一样用，需要进行额外的提取

生成多跳双星图的谓词和示例规则，用于生成式模型学习并生成示例规则
𝑟 ::= 𝑇(𝑦, 𝑡)| 𝑇(𝑓𝑦,𝑡)| M(𝑦.𝐵, 𝑦.¯ 𝐶¯)| M(𝑓𝑦.𝐵, 𝑓¯ 𝑦.𝐶¯)|
𝑦.𝐵 ⊕ 𝑦.𝐶 | 𝑨.𝐵 ⊕ 𝑑 | 𝑓𝑦.𝐵 ⊕ 𝑓𝑦.𝐶 | 𝑓𝑨.𝐵 ⊕ 𝑑 |
𝑈(𝑡, 𝑡′,𝜐)| 𝜐(𝑦.𝐵, 𝑦.𝐶)| 𝜐(𝑓𝑦.𝐵, 𝑓𝑦.𝐶).
"""

import os
from glob import glob
import json
import random
import re
from collections import defaultdict
from itertools import combinations
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 初始化LLM客户端
llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


# Model 1: Compute Conflict Score (Contradiction/Entailment)
def M_predicate_new(attrs1: dict, attrs2: dict, x, y, ml_threshold: float = 0.5) -> str:
    """
    Given two sets of attributes, calculate the highest score from three models:
    1) Conflict score (contradiction/entailment)
    2) Contextual dependency and co-occurrence
    3) Temporal inference

    Returns a label indicating the highest probability.
    """
    # Extract relevant textual information from the attributes
    premise = ' '.join(f"{key}: {value}" for key, value in attrs1.items() if key not in ['neighbors', 'multi_hop_neighbors'])
    hypothesis = ' '.join(f"{key}: {value}" for key, value in attrs2.items() if key not in ['neighbors', 'multi_hop_neighbors'])

    # Combined prompt for all three models
    system_message = (
        "You are a multi-task assistant capable of evaluating the following:\n"
        "1) Conflict score — Determine the probability of contradiction between two sentences.\n"
        "2) Contextual Dependency and Co-occurrence — Evaluate how much the meaning of one sentence depends on the other and how often they co-occur.\n"
        "3) Temporal Inference — Evaluate whether one event happens before another.\n"
        "You will receive two sentences. For each of the tasks above, provide a score between 0 and 1."
        "\nYour output will be a single string with the following structure:\n"
        "'<conflict_score>, <dependency_score>, <co_occurrence_score>, <temporal_score>'\n"
        "Where each score is a decimal value between 0 and 1."
    )

    user_message = f"""
    Premise: "{premise}"
    Hypothesis: "{hypothesis}"
    """

    response = llm.generate(system_message, user_message)
    print("LLM Response:", response)

    try:
        # Parse the response into individual scores
        conflict_score, dep_score, cooc_score, temporal_score = map(float, response.strip().split(","))

        # Clamp the values to be between 0 and 1
        conflict_score = max(0.0, min(1.0, conflict_score))
        dep_score = max(0.0, min(1.0, dep_score))
        cooc_score = max(0.0, min(1.0, cooc_score))
        temporal_score = max(0.0, min(1.0, temporal_score))
    except Exception as e:
        print("Error parsing the model response, defaulting to zero values.", e)
        conflict_score = dep_score = cooc_score = temporal_score = 0.0

    # Select the highest score and model type
    scores = {
        "conflict": conflict_score,
        "dependency": dep_score,
        "co-occurrence": cooc_score,
        "temporal": temporal_score
    }

    highest_score_model = max(scores, key=scores.get)
    highest_score = scores[highest_score_model]

    def get_identifier(attrs: dict):
        # Take the first key-value pair if no name exists
        return f"({', '.join([f'{k}: {v}' for k, v in attrs.items()])})"

    # identifier_1 = get_identifier(attrs1)
    # identifier_2 = get_identifier(attrs2)
    identifier_1 = x + ".attrs"
    identifier_2 = y + ".attrs"

    # Return the result based on the highest scoring model
    if highest_score >= ml_threshold:
        if highest_score_model == "conflict":
            return f"M_NLI({identifier_1}, {identifier_2})"
        elif highest_score_model == "dependency":
            return f"M_NLU({identifier_1}, {identifier_2})"
        elif highest_score_model == "temporal":
            return f"M_TEM({identifier_1}, {identifier_2})"
    else:
        return f"None_ML({identifier_1}, {identifier_2})"


def detect_and_classify_contradiction(sentence1, sentence2, llm):
    """
    输入两个句子，调用LLM一次性判断是否矛盾，若矛盾则分类矛盾类型并定位。
    返回：
      - 如果矛盾，返回 dict: {"type": "...", "loc1": "...", "loc2": "...", "explanation": "..."}
      - 如果不矛盾，返回: {"type": "None", "loc1": null, "loc2": null, "explanation": null}
    """
    prompt = f"""
You are an expert in contradiction detection and analysis.

Given two sentences:
- Determine whether they contradict.
- If they do, classify the contradiction into one of: ["Factual", "Content", "Causal", "Negation", "Relation", "Numeric", "Temporal", "Violation"].
- Identify the two conflicting attributes (either node or edge properties), in the format like x.A or ex.A.
- If they are not contradictory, set type to "None", and loc1, loc2, explanation to null.

Output your answer **only** in this exact JSON format:
{{
  "type": "<contradiction_type or 'None'>",
  "loc1": "<attribute1 or null>",
  "loc2": "<attribute2 or null>",
  "explanation": "<explanation text or null>"
}}

Now analyze the following:

Sentence 1: "{sentence1}"
Sentence 2: "{sentence2}"
"""
    if not sentence1 or not sentence2:
        print("⚠️ 输入的句子无效，跳过")
        return {"type": "Unknown", "loc1": None, "loc2": None, "explanation": "Invalid input sentences."}

    system_prompt = "You are a reasoning engine that outputs structured JSON responses for contradiction detection."

    response = llm.generate(
        user_message=prompt,
        system_message=system_prompt
    ).strip()

    import json
    try:
        result = json.loads(response)
    except Exception:
        return {
            "type": "Unknown",
            "loc1": None,
            "loc2": None,
            "explanation": f"[Unparsable JSON] Raw LLM output:\n{response}"
        }

    # fallback defaults
    result.setdefault("type", "Unknown")
    result.setdefault("loc1", None)
    result.setdefault("loc2", None)
    result.setdefault("explanation", None)

    return result


def gen_contradiction_type_preds(predicates, llm, max_t_predicates=8):
    import re
    import random
    from itertools import combinations

    s_strings = []
    s_mapping = {}  # 记录 span -> 实体/属性 名称映射（用于反查 x.A、ex.A）

    pattern = re.compile(r'S\(([^,]+),\s*"([^"]+)"\)')  # 匹配 S(x, "span")

    for p in predicates:
        m = pattern.match(p)
        if m:
            key = m.group(2)  # span文本
            obj = m.group(1)  # x 或 ex
            s_strings.append(key)
            s_mapping[key] = obj  # 映射 span -> 实体/边名

    # 生成所有两两组合并打乱
    pairs = list(combinations(s_strings, 2))
    random.shuffle(pairs)

    count = 0
    for s1, s2 in pairs:
        if count >= max_t_predicates:
            break

        contradiction_info = detect_and_classify_contradiction(s1, s2, llm)
        print("🔍 contradiction_info:", contradiction_info)

        # 提取实体或边的名字
        x = s_mapping.get(s1, "x")
        y = s_mapping.get(s2, "y")

        if not contradiction_info:
            continue  # 出错跳过

        if contradiction_info.get("type") == "None":
            predicates.add(f"None({x}, {y})")
            predicates.add(f"T({repr(s1)}, {repr(s2)}, {repr('None')})")
            continue  # 不继续执行后续逻辑

        ctype = contradiction_info.get("type")
        loc1 = contradiction_info.get("loc1")
        loc2 = contradiction_info.get("loc2")

        predicates.add(f"T({repr(s1)}, {repr(s2)}, {repr(ctype)})")

        if ctype == "Violation":  # 如果是违反类型的矛盾
            predicates.add(f"violate({x}, {y})")

        if loc1 and loc2:
            predicates.add(f"{ctype}({loc1}, {loc2})")

        count += 1



def gen_span_mapping_preds(predicates, max_s_predicates=8):
    """
    将节点和边与其在原始文档中的子字符串映射为谓词 S(x, s) 或 S(ex, s)。
    可选参数 max_s_predicates 用于限制总数，节点和边均分。
    节点和边部分将从数据中随机抽取。
    """

    if max_s_predicates is not None:
        max_entity_s = max_s_predicates // 2
        max_edge_s = max_s_predicates - max_entity_s
    else:
        max_entity_s = max_edge_s = None

    entity_count = 0
    edge_count = 0

    def maybe_add_entity_s(predicate_str):
        nonlocal entity_count
        if max_entity_s is not None and entity_count >= max_entity_s:
            return False
        predicates.add(predicate_str)
        entity_count += 1
        return True

    def maybe_add_edge_s(predicate_str):
        nonlocal edge_count
        if max_edge_s is not None and edge_count >= max_edge_s:
            return False
        predicates.add(predicate_str)
        edge_count += 1
        return True

    # 加载实体 span 映射（随机）
    all_entity_entries = []
    for entity_file in ["entity_text1.json", "entity_text2.json"]:
        try:
            with open(os.path.join(OUTPUT_DIR, entity_file), "r", encoding="utf-8") as f:
                entries = json.load(f)
                all_entity_entries.extend(entries)
        except Exception as e:
            print(f"⚠️ 加载实体span失败：{entity_file} - {e}")

    random.shuffle(all_entity_entries)

    for entry in all_entity_entries:
        node1 = entry["triple"]["node_1"]
        node2 = entry["triple"]["node_2"]
        span1 = entry["spans"].get("node_1_span", "").strip().replace('"', "'")
        span2 = entry["spans"].get("node_2_span", "").strip().replace('"', "'")

        if node1 and span1:
            node1_name = node1["name"].replace(" ", "_")
            if not maybe_add_entity_s(f'S({node1_name}, "{span1}")'):
                break

        if node2 and span2:
            node2_name = node2["name"].replace(" ", "_")
            if not maybe_add_entity_s(f'S({node2_name}, "{span2}")'):
                break

    # 加载边 span 映射（随机）
    all_edge_entries = []
    for edge_file in ["edge_text1.json", "edge_text2.json"]:
        try:
            with open(os.path.join(OUTPUT_DIR, edge_file), "r", encoding="utf-8") as f:
                entries = json.load(f)
                all_edge_entries.extend(entries)
        except Exception as e:
            print(f"⚠️ 加载边span失败：{edge_file} - {e}")

    random.shuffle(all_edge_entries)

    for entry in all_edge_entries:
        triple = entry["triple"]
        span = entry.get("relation_context_span", "").strip()
        if span.startswith('"') and span.endswith('"'):
            span = span[1:-1]
        span = span.replace('"', "'")

        if not span:
            continue

        edge_name = triple.get("relationship")
        if not maybe_add_edge_s(f'S({edge_name}, "{span}")'):
            break

def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_dual_star_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_document_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

from itertools import combinations
from collections import defaultdict
import random

def generate_predicates_and_rules(data, doc1, doc2, x0, y0, ml_threshold=0.5):
    predicates = set()
    rules = []

    nodes_attrs = {}  # This will store all the node attributes
    edges_attrs = {}
    text_spans = {}

    # 处理双星结构中心节点
    for entry in data:
        center = entry["center"]

        # 节点属性：包括 neighbors 和 multi_hop_neighbors
        neighbors = entry["neighbors"]
        multi_hop_neighbors = entry["multi_hop_neighbors"]

        # 节点属性存储
        nodes_attrs.setdefault(center, {"neighbors": neighbors, "multi_hop_neighbors": multi_hop_neighbors})

        # 关系和边属性处理
        for neighbor, edge in zip(neighbors, entry["edges"]):
            edge_key = (center, neighbor)
            edges_attrs.setdefault(edge_key, {})
            edges_attrs[edge_key]["relationship"] = edge  # 使用边的描述作为关系

            # 如果有多跳邻居，加入到边属性中
            for hop in multi_hop_neighbors:
                if hop["neighbor"] == neighbor:
                    edges_attrs[edge_key]["multi_hop"] = hop

        # 文本跨度
        if "text_span" in entry:
            text_spans[center] = entry["text_span"]

        # 处理多跳节点和多跳边的属性
        for hop in multi_hop_neighbors:
            hop_node = hop["neighbor"]
            if hop_node not in nodes_attrs:
                nodes_attrs[hop_node] = {}  # 初始化多跳节点属性
            hop_edge = hop["edge"]
            edges_attrs[(center, hop_node)] = {"relationship": hop_edge}  # 为多跳边生成边属性

            # 补充多跳边的属性
            if hop_edge in entry["multi_hop_edges_attrs"]:
                edges_attrs[(center, hop_node)].update(entry["multi_hop_edges_attrs"][hop_edge])

            # 更新多跳节点的属性
            # if hop_node in entry["multi_hop_node_attrs"]:
            #     nodes_attrs[hop_node].update(entry["multi_hop_node_attrs"][hop_node])
            multi_hop_node_attrs = entry.get("multi_hop_node_attrs", {})
            if hop_node in multi_hop_node_attrs:
                nodes_attrs[hop_node].update(multi_hop_node_attrs[hop_node])

        # 合并 node_attrs
        if "node_attrs" in entry:
            for node, attrs in entry["node_attrs"].items():
                if node not in nodes_attrs:
                    nodes_attrs[node] = {}
                nodes_attrs[node].update(attrs)

        # 合并 edge_attrs
        if "edges_attrs" in entry:
            for edge, attrs in entry["edges_attrs"].items():
                if edge not in edges_attrs:
                    edges_attrs[edge] = {}
                edges_attrs[edge].update(attrs)

    # --------------------谓词生成---------------------

    def gen_unary_node_preds():
        for node, attrs in nodes_attrs.items():
            for attr, val in attrs.items():
                if isinstance(val, (int, float)):
                    predicates.add(f"{node}.{attr} = {val}")
                    predicates.add(f"{node}.{attr} != {val}")
                    predicates.add(f"{node}.{attr} < {val + 5}")
                    predicates.add(f"{node}.{attr} > {val - 5}")
                    predicates.add(f"{node}.{attr} <= {val}")
                    predicates.add(f"{node}.{attr} >= {val}")
                elif isinstance(val, str):
                    predicates.add(f'{node}.{attr} = "{val}"')
                    predicates.add(f'{node}.{attr} != "{val}"')

    def gen_binary_node_preds():
        nodes = list(nodes_attrs.keys())
        for x, y in combinations(nodes, 2):
            attrs_x = nodes_attrs[x]
            attrs_y = nodes_attrs[y]
            for attr in set(attrs_x) & set(attrs_y):
                if type(attrs_x[attr]) != type(attrs_y[attr]):
                    continue
                if isinstance(attrs_x[attr], (int, float)):
                    predicates.add(f"{x}.{attr} != {y}.{attr}")
                elif isinstance(attrs_x[attr], str):
                    predicates.add(f'{x}.{attr} != {y}.{attr}')

    def gen_unary_edge_preds():
        """
        生成边的单一属性谓词，处理边的所有属性，包括关系（relationship）和其他属性。
        """
        for key, attrs in edges_attrs.items():
            try:
                # 确保 key 是一个包含两个元素的元组 (x, y)
                if isinstance(key, tuple) and len(key) == 2:
                    x, y = key
                else:
                    # 如果 key 不是元组或者元组长度不为 2，则跳过当前条目
                    print(f"Skipping invalid edge key: {key}. Expected a tuple of length 2.")
                    continue  # 跳过格式不正确的边数据

                for attr, val in attrs.items():
                    # 跳过 text_span 字段
                    if attr == "text_span":
                        continue

                    # 处理数值属性
                    if isinstance(val, (int, float)):
                        predicates.add(f"{x}___{y}.{attr} = {val}")
                        predicates.add(f"{x}___{y}.{attr} != {val}")
                        predicates.add(f"{x}___{y}.{attr} < {val + 1}")
                        predicates.add(f"{x}___{y}.{attr} > {val - 1}")
                        predicates.add(f"{x}___{y}.{attr} <= {val}")
                        predicates.add(f"{x}___{y}.{attr} >= {val}")

                    # 处理字符串属性（例如关系，地点，话题等）
                    elif isinstance(val, str):
                        predicates.add(f'{x}___{y}.{attr} = "{val}"')
                        predicates.add(f'{x}___{y}.{attr} != "{val}"')

                    # 处理列表类型属性（如多跳边）
                    elif isinstance(val, list):
                        predicates.add(f"{x}___{y}.{attr}.length = {len(val)}")
                        # 处理列表中的元素（例如每个多跳边）
                        for i, item in enumerate(val):
                            if isinstance(item, dict):
                                for sub_key, sub_val in item.items():
                                    if isinstance(sub_val, (int, float)):
                                        predicates.add(f"{x}___{y}.{attr}.{i}.{sub_key} = {sub_val}")
                                    elif isinstance(sub_val, str):
                                        predicates.add(f'{x}___{y}.{attr}.{i}.{sub_key} = "{sub_val}"')

                # 处理多跳边 (multi_hop)
                if "multi_hop" in attrs:
                    for hop in attrs["multi_hop"]:
                        try:
                            # 假设 hop 是一个字典，包含 'neighbor', 'edge', 和 'hop_count'
                            predicates.add(f"{x}___{y}_multi_hop.{hop['neighbor']}.{hop['edge']} = {hop['hop_count']}")
                        except KeyError as e:
                            print(f"KeyError in multi_hop: {e}. Skipping this hop entry.")
                            continue  # 跳过当前 hop

            except Exception as e:
                print(f"An error occurred while processing edge ({x}, {y}): {e}. Skipping this edge.")
                continue  # 跳过当前边

    def gen_binary_edge_preds():
        """
        生成边的二元属性谓词，主要是比较不同边之间的属性（如关系是否相同等）。
        """
        # 确保所有边的键都是有效的二元组
        valid_edges = []
        for key in edges_attrs.keys():
            if isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, str) for k in key):
                valid_edges.append(key)
            else:
                print(f"Skipping invalid edge key: {key}. Expected a tuple of length 2 with string elements.")

        # 获取边列表的所有组合
        for (ex1, ey1), (ex2, ey2) in combinations(valid_edges, 2):
            try:
                # 获取两条边的关系属性
                attr1 = edges_attrs.get((ex1, ey1), {}).get("relationship", None)
                attr2 = edges_attrs.get((ex2, ey2), {}).get("relationship", None)

                # 如果关系不同，添加谓词
                if attr1 != attr2:
                    predicates.add(f'{ex1}___{ey1}.relationship != {ex2}_{ey2}.relationship')

                # 比较其他属性（关系以外的）
                common_attrs = set(edges_attrs.get((ex1, ey1), {}).keys()) & set(edges_attrs.get((ex2, ey2), {}).keys())
                for attr in common_attrs:
                    if attr != "relationship":  # 忽略已处理的关系属性
                        val1 = edges_attrs[(ex1, ey1)].get(attr)
                        val2 = edges_attrs[(ex2, ey2)].get(attr)
                        # 如果属性值不同，添加谓词
                        if val1 != val2:
                            predicates.add(f'{ex1}_{ey1}.{attr} != {ex2}_{ey2}.{attr}')
            except Exception as e:
                print(f"Error processing edge ({ex1}, {ey1}) and ({ex2}, {ey2}): {e}. Skipping this comparison.")
                continue

    def print_predicate_statistics(predicates):
        stats = defaultdict(int)
        for p in predicates:
            if "M_NLI" in p or "M_NLU" in p or "M_TEM" in p:
                stats["ML"] += 1
            elif "relationship" in p or "_.relationship" in p:
                stats["edge_attr"] += 1
            else:
                stats["entity_attr"] += 1

        print("📊 Predicate Type Statistics:")
        for k, v in stats.items():
            print(f" - {k}: {v}")

    def gen_ml_preds(m=5, n=5):
        """
        Generate `m` random combinations of node predicates and `n` random combinations of edge predicates.

        :param m: Number of random node combinations to generate
        :param n: Number of random edge combinations to generate
        """
        # 随机选择 m 个节点对
        nodes = list(nodes_attrs.keys())
        node_combinations = random.sample(list(combinations(nodes, 2)), m)

        for x, y in node_combinations:
            # 对每对节点生成谓词
            score = M_predicate_new(nodes_attrs[x], nodes_attrs[y], x, y)
            predicates.add(score)

        # 随机选择 n 个边对
        edges = list(edges_attrs.keys())
        edge_combinations = random.sample(list(combinations(edges, 2)), n)

        # 处理随机选择的边组合
        for ex, ey in edge_combinations:
            # 如果边是元组类型（包含两个节点），处理这些边的属性
            if isinstance(ex, tuple) and isinstance(ey, tuple):
                ex_attrs = edges_attrs.get(ex, {})
                ey_attrs = edges_attrs.get(ey, {})
            else:
                # 如果边是字符串类型（单边边缘），需要确保其属性已经存在
                ex_attrs = edges_attrs.get(ex, {})
                ey_attrs = edges_attrs.get(ey, {})

            # 如果边的属性为空，跳过该组合
            if not ex_attrs or not ey_attrs:
                continue

            # 确保生成的 score 字符串正确连接
            ex_str = f"{ex}" if isinstance(ex, tuple) else ex
            ey_str = f"{ey}" if isinstance(ey, tuple) else ey

            # 调用 M_predicate_new 函数生成边预测
            score = M_predicate_new(ex_attrs, ey_attrs, ex_str, ey_str)

            # 添加生成的预测分数
            predicates.add(score)


    # 执行所有谓词生成器
    gen_unary_node_preds()
    gen_binary_node_preds()
    gen_unary_edge_preds()
    gen_binary_edge_preds()
    gen_ml_preds()
    gen_span_mapping_preds(predicates)  # ← 添加到这里
    # 新增：基于S谓词生成T(s,s',τ)矛盾类型谓词
    gen_contradiction_type_preds(predicates, llm)

    # 假设 predicates 已有
    predicates = {p.strip().replace("  ", " ") for p in predicates if p}
    print_predicate_statistics(predicates)
    pred_list = list(predicates)
    rules = []
    rule_count = 0
    max_rules = 10

    # 先按类型分类谓词
    type_buckets = defaultdict(list)
    for p in pred_list:
        if "violate" in p:
            type_buckets["violate"].append(p)
        elif "M(" in p:
            type_buckets["M"].append(p)
        else:
            type_buckets["other"].append(p)

    # 确保每个类型至少出现一次作为后件
    used_heads = set()
    for t, preds_of_type in type_buckets.items():
        head = random.choice(preds_of_type)
        used_heads.add(head)
        bodies = [p for p in pred_list if p != head]
        if not bodies:
            continue
        body = random.choice(bodies)
        rule_body = f"{body} ∧ {head}"
        rules.append(f"Q[{x0},{y0}] ({rule_body} → {head})")
        rule_count += 1
        if rule_count >= max_rules:
            break

    # 生成剩余规则，随机后件，但不重复用已经用过的head，保证多样性
    combos = list(combinations(pred_list, 2))
    random.shuffle(combos)
    for combo in combos:
        body = " ∧ ".join(combo)
        head = random.choice(combo)
        if head in used_heads:
            continue  # 跳过重复的后件
        rule = f"Q[{x0},{y0}] ({body} → {head})"
        rules.append(rule)
        used_heads.add(head)
        rule_count += 1
        if rule_count >= max_rules:
            break

    return predicates, rules




def find_dual_star_centers(data):
    """
    从给定的数据中，查找并返回两个中心节点。
    假设每个数据块都有一个 `center` 字段，表示该中心节点的名称。
    返回两个中心节点的名称 x0 和 y0。
    """
    centers = []

    # 遍历数据，提取 center 节点
    for entry in data:
        center = entry.get("center")
        if center:
            centers.append(center)

        # 如果找到了两个中心节点，直接返回
        if len(centers) == 2:
            return centers[0], centers[1]

    raise ValueError("❌ 未找到两个中心节点")


if __name__ == "__main__":
    # 设定你的目录
    DUAL_STAR_JSON_DIR = "./dual_star_json_with_attrs"
    OUTPUT_DIR = "dual_star"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 遍历所有 JSON 文件
    json_files = glob(os.path.join(DUAL_STAR_JSON_DIR, "*.json"))
    # 取前三个json
    json_files = json_files[:3]

    for json_path in json_files:
        filename = os.path.basename(json_path)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"\n🌀 处理中: {filename}")

        try:
            # 加载 JSON 数据
            data = load_dual_star_json(json_path)
            x0, y0 = find_dual_star_centers(data)
            text1_path = "text1.txt"
            text2_path = "text2.txt"

            if not os.path.exists(text1_path) or not os.path.exists(text2_path):
                print(f"⚠️ 缺失对应文本文件: {text1_path} 或 {text2_path}")
                continue

            doc1 = load_text_file(text1_path)
            doc2 = load_text_file(text2_path)

            preds, rules = generate_predicates_and_rules(data, doc1, doc2, x0, y0)

            # 输出文件
            output_file = os.path.join(OUTPUT_DIR, f"{name_without_ext}_rico.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("Predicates:\n")
                for p in sorted(preds):
                    f.write(p + "\n")

                f.write("\nRules:\n")
                for r in rules:
                    f.write(r + "\n")

            print(f"✅ 已保存到: {output_file}")

        except Exception as e:
            print(f"❌ 处理失败: {filename} - 错误: {e}")
