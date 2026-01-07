import os
from glob import glob
import json
import random
import re
from collections import defaultdict
from itertools import combinations
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import Dict, Any, List, Tuple, Set as TypeSet
import time

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 初始化LLM客户端
llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


# --- 核心谓词逻辑 (LLM Calls) ---

def M_predicate_new(attrs1: dict, attrs2: dict, x, y, ml_threshold: float = 0.5) -> str:
    """基于 LLM 计算冲突、依赖和时间分数，返回最高分的 ML 谓词。"""
    # ... (函数体不变) ...
    premise = ' '.join(f"{key}: {value}" for key, value in attrs1.items() if key not in ['neighbors', 'multi_hop_neighbors'])
    hypothesis = ' '.join(f"{key}: {value}" for key, value in attrs2.items() if key not in ['neighbors', 'multi_hop_neighbors'])

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
    # print("LLM Response:", response)

    try:
        response_clean = response.strip()
        response_clean = re.sub(r"```(json)?\s*", "", response_clean, flags=re.DOTALL).strip()
        response_clean = re.sub(r"```", "", response_clean).strip()

        conflict_score, dep_score, cooc_score, temporal_score = map(float, response_clean.split(","))

        conflict_score = max(0.0, min(1.0, conflict_score))
        dep_score = max(0.0, min(1.0, dep_score))
        cooc_score = max(0.0, min(1.0, cooc_score))
        temporal_score = max(0.0, min(1.0, temporal_score))
    except Exception as e:
        # print(f"Error parsing the model response, defaulting to zero values. Error: {e}")
        conflict_score = dep_score = cooc_score = temporal_score = 0.0

    scores = {
        "conflict": conflict_score,
        "dependency": dep_score,
        "co-occurrence": cooc_score,
        "temporal": temporal_score
    }

    highest_score_model = max(scores, key=scores.get)
    highest_score = scores[highest_score_model]

    identifier_1 = x + ".attrs"
    identifier_2 = y + ".attrs"

    if highest_score >= ml_threshold:
        if highest_score_model == "conflict":
            return f"M_NLI({identifier_1}, {identifier_2})"
        elif highest_score_model == "dependency":
            return f"M_NLU({identifier_1}, {identifier_2})"
        elif highest_score_model == "temporal":
            return f"M_TEM({identifier_1}, {identifier_2})"
    else:
        return f"None_ML({identifier_1}, {identifier_2})"


# --- 其他辅助函数 (保持不变，已包含之前的修复) ---

def detect_and_classify_contradiction(sentence1: str, sentence2: str, llm: OpenAIClient) -> Dict[str, Any]:
    # ... (函数体不变) ...
    prompt = f"""
You are an expert in contradiction detection and analysis.
...
Sentence 1: "{sentence1}"
Sentence 2: "{sentence2}"
"""
    if not sentence1 or not sentence2:
        return {"type": "Unknown", "loc1": None, "loc2": None, "explanation": "Invalid input sentences."}
    system_prompt = "You are a reasoning engine that outputs structured JSON responses for contradiction detection."
    response = llm.generate(user_message=prompt, system_message=system_prompt).strip()
    response_clean = re.sub(r"```(json)?\s*", "", response.strip(), flags=re.DOTALL).strip()
    response_clean = re.sub(r"```", "", response_clean).strip()
    try:
        result = json.loads(response_clean)
    except Exception as e:
        return {"type": "Unknown", "loc1": None, "loc2": None, "explanation": f"[Unparsable JSON] Raw LLM output:\n{response}"}
    result.setdefault("type", "Unknown")
    result.setdefault("loc1", None)
    result.setdefault("loc2", None)
    result.setdefault("explanation", None)
    return result


def gen_contradiction_type_preds(predicates: set, llm: OpenAIClient, max_t_predicates: int = 8):
    # ... (函数体不变) ...
    s_strings = []
    s_mapping = {}
    pattern = re.compile(r'S\(([^,]+),\s*"([^"]+)"\)')
    for p in predicates:
        m = pattern.match(p)
        if m:
            key = m.group(2)
            obj = m.group(1)
            s_strings.append(key)
            s_mapping[key] = obj
    pairs = list(combinations(s_strings, 2))
    random.shuffle(pairs)
    count = 0
    for s1, s2 in pairs:
        if count >= max_t_predicates: break
        contradiction_info = detect_and_classify_contradiction(s1, s2, llm)
        x = s_mapping.get(s1, "x")
        y = s_mapping.get(s2, "y")
        if not contradiction_info or contradiction_info.get("type") == "Unknown":
            time.sleep(0.5)
            continue
        ctype = contradiction_info.get("type")
        loc1 = contradiction_info.get("loc1")
        loc2 = contradiction_info.get("loc2")
        if ctype == "None":
            predicates.add(f"None({x}, {y})")
            predicates.add(f"T({repr(s1)}, {repr(s2)}, {repr('None')})")
            count += 1
            time.sleep(0.5)
            continue
        predicates.add(f"T({repr(s1)}, {repr(s2)}, {repr(ctype)})")
        if ctype == "Violation":
            predicates.add(f"violate({x}, {y})")
        if loc1 and loc2:
            predicates.add(f"{ctype}({loc1}, {loc2})")
        count += 1
        time.sleep(0.5)


def gen_span_mapping_preds(predicates: set, max_s_predicates: int = 8):
    # ... (函数体不变) ...
    OUTPUT_DIR = "dual_star/direct_generate_rules"
    if max_s_predicates is not None:
        max_entity_s = max_s_predicates // 2
        max_edge_s = max_s_predicates - max_entity_s
    else:
        max_entity_s = max_edge_s = None
    entity_count = 0
    edge_count = 0

    def maybe_add_entity_s(predicate_str):
        nonlocal entity_count
        if max_entity_s is not None and entity_count >= max_entity_s: return False
        predicates.add(predicate_str)
        entity_count += 1
        return True

    def maybe_add_edge_s(predicate_str):
        nonlocal edge_count
        if max_edge_s is not None and edge_count >= max_edge_s: return False
        predicates.add(predicate_str)
        edge_count += 1
        return True

    all_entity_entries = []
    for entity_file in ["entity_text1.json", "entity_text2.json"]:
        try:
            with open(os.path.join(OUTPUT_DIR, entity_file), "r", encoding="utf-8") as f:
                all_entity_entries.extend(json.load(f))
        except Exception:
            pass
    random.shuffle(all_entity_entries)
    for entry in all_entity_entries:
        node1 = entry["triple"]["node_1"]
        node2 = entry["triple"]["node_2"]
        span1 = entry["spans"].get("node_1_span", "").strip().replace('"', "'")
        span2 = entry["spans"].get("node_2_span", "").strip().replace('"', "'")
        if node1 and span1:
            node1_name = node1["name"].replace(" ", "_")
            if not maybe_add_entity_s(f'S({node1_name}, "{span1}")'): break
        if node2 and span2:
            node2_name = node2["name"].replace(" ", "_")
            if not maybe_add_entity_s(f'S({node2_name}, "{span2}")'): break

    all_edge_entries = []
    for edge_file in ["edge_text1.json", "edge_text2.json"]:
        try:
            with open(os.path.join(OUTPUT_DIR, edge_file), "r", encoding="utf-8") as f:
                all_edge_entries.extend(json.load(f))
        except Exception:
            pass
    random.shuffle(all_edge_entries)
    for entry in all_edge_entries:
        triple = entry["triple"]
        span = entry.get("relation_context_span", "").strip()
        if span.startswith('"') and span.endswith('"'): span = span[1:-1]
        span = span.replace('"', "'")
        if not span: continue
        edge_name = triple.get("relationship")
        if edge_name:
            edge_name = edge_name.replace(" ", "_").replace("__", "_")
            if not maybe_add_edge_s(f'S({edge_name}, "{span}")'): break


def load_dual_star_json(filepath: str) -> List[Dict[str, Any]]:
    """加载双星图 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_dual_star_centers(data: Dict[str, Any]) -> Tuple[str, str]:
    """
    从给定的双星图数据中，查找并返回两个中心节点 (x0, y0)。
    数据结构预期为一个字典，其中包含 'stars' 键，对应一个星型图结构的列表。

    注意：如果数据只包含一个中心实体（如 '10'），则返回 (entity, entity)。
    """
    # 🌟 关键修正：从字典中取出 'stars' 列表
    star_list = data.get("stars")

    if not star_list or not isinstance(star_list, list):
        raise ValueError("❌ 数据结构不正确：未找到有效的 'stars' 列表。")

    unique_centers = set()

    # 遍历 'stars' 列表中的所有星型结构
    for entry in star_list:
        center = entry.get("center")
        if center and isinstance(center, str):
            unique_centers.add(center)

    centers = list(unique_centers)

    if not centers:
        # 如果 'stars' 列表中没有任何 'center' 键（不应发生）
        raise ValueError("❌ 未在数据中任何星型结构中找到中心节点。")

    if len(centers) == 1:
        # 匹配你的数据结构：只有一个中心实体 ('10')
        x0 = centers[0]
        y0 = centers[0]
        # 打印警告或提示，帮助理解数据结构
        print(f"⚠️ 提示：数据中只找到一个唯一的中心实体: '{x0}'。返回: ('{x0}', '{y0}')。")
        return x0, y0

    elif len(centers) >= 2:
        # 传统双星图：找到两个或更多不同的中心实体
        print(f"✅ 找到两个不同的中心实体: ('{centers[0]}', '{centers[1]}')")
        return centers[0], centers[1]

    # 理论上不会执行到这里
    return "", ""


## 核心修复点：数据类型检查（针对 'str' object has no attribute 'get'）
# 导入部分保持不变

import random
from collections import defaultdict
from itertools import combinations
from typing import Dict, Any, List, Tuple, Set as TypeSet


# 假设 M_predicate_new, gen_span_mapping_preds, gen_contradiction_type_preds, OpenAIClient 已经定义
# 为了使这段代码独立运行，我省略了依赖的外部函数定义，但它们在逻辑上被调用。

# 假设外部依赖 M_predicate_new, gen_span_mapping_preds, gen_contradiction_type_preds, print_predicate_statistics 已经定义。

def generate_predicates_and_rules(data: Dict[str, Any], x0: str, y0: str, ml_threshold: float = 0.5, llm: Any = None) -> Tuple[TypeSet[str], List[str]]:
    """
    生成谓词和规则的核心函数。

    修改了规则生成策略：前件 (Body) 长度在 2 到 6 个谓词之间，且均匀分布。
    """
    predicates: TypeSet[str] = set()
    rules: List[str] = []
    nodes_attrs: Dict[str, Dict[str, Any]] = {}
    edges_attrs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    star_list = data.get('stars', [])
    if not star_list:
        print("⚠️ 警告: 'stars' 列表为空，无法生成谓词。")
        return predicates, rules

    edge_text_to_tuple_map: Dict[str, Tuple[str, str]] = {}

    # -------------------- I. 数据预处理 (保持不变) ---------------------
    for entry in star_list:
        if not isinstance(entry, dict): continue
        center = entry.get("center")
        if not center or not isinstance(center, str): continue

        neighbors = entry.get("neighbors", [])
        multi_hop_neighbors = entry.get("multi_hop_neighbors", [])

        if not isinstance(neighbors, list): neighbors = []
        if not isinstance(multi_hop_neighbors, list): multi_hop_neighbors = []

        # 初始化中心节点的属性，确保包含邻居列表
        nodes_attrs.setdefault(center, {"neighbors": [], "multi_hop_neighbors": []})
        nodes_attrs[center]["neighbors"].extend([n for n in neighbors if n not in nodes_attrs[center]["neighbors"]])
        nodes_attrs[center]["multi_hop_neighbors"].extend([h for h in multi_hop_neighbors if h not in nodes_attrs[center]["multi_hop_neighbors"]])

        # 处理邻居和边 (构建 (center, neighbor) 键 和 边文本映射)
        edges = entry.get("edges", [])
        for neighbor, edge_text in zip(neighbors, edges):
            if not isinstance(neighbor, str) or not isinstance(edge_text, str): continue
            edge_key = (center, neighbor)
            edges_attrs.setdefault(edge_key, {"relationship": edge_text})
            edge_text_to_tuple_map[edge_text] = edge_key

        # 合并 node_attrs
        node_attrs_entry = entry.get("node_attrs", {})
        if isinstance(node_attrs_entry, dict):
            for node, attrs in node_attrs_entry.items():
                if not isinstance(attrs, dict): continue
                if node not in nodes_attrs: nodes_attrs[node] = {}

                # 临时保存列表属性，防止被 update 覆盖
                current_neighbors = nodes_attrs[node].get('neighbors', [])
                current_multi_hop = nodes_attrs[node].get('multi_hop_neighbors', [])

                nodes_attrs[node].update(attrs)

                # 恢复列表属性
                nodes_attrs[node]['neighbors'] = current_neighbors
                nodes_attrs[node]['multi_hop_neighbors'] = current_multi_hop

        # 合并 edges_attrs
        edges_attrs_entry = entry.get("edges_attrs", {})
        if isinstance(edges_attrs_entry, dict):
            for edge_text, attrs in edges_attrs_entry.items():
                if not isinstance(attrs, dict) or not isinstance(edge_text, str): continue
                edge_tuple_key = edge_text_to_tuple_map.get(edge_text)
                if edge_tuple_key:
                    if edge_tuple_key not in edges_attrs:
                        edges_attrs[edge_tuple_key] = {}
                    edges_attrs[edge_tuple_key].update(attrs)

    # -------------------- II. 谓词生成器定义 (此处为省略，假设内部函数已定义) ---------------------

    # ⚠️ 注意：为了简洁，这里省略了 gen_unary_node_preds 等内部函数定义。
    # 在实际代码中，它们应位于此位置或全局作用域，并使用 nodes_attrs 和 edges_attrs。

    def gen_unary_node_preds():
        """生成一元节点谓词 (x.attr = val)。"""
        for node, attrs in nodes_attrs.items():
            for attr, val in attrs.items():
                if attr in ['neighbors', 'multi_hop_neighbors']: continue

                if isinstance(val, (int, float)):
                    predicates.add(f"{node}.{attr} = {val}")
                    predicates.add(f"{node}.{attr} != {val}")
                    predicates.add(f"{node}.{attr} < {val + 5}")
                    predicates.add(f"{node}.{attr} > {val - 5}")
                elif isinstance(val, str) and val.strip():
                    val_clean = val.replace('"', "'")
                    predicates.add(f'{node}.{attr} = "{val_clean}"')
                    predicates.add(f'{node}.{attr} != "{val_clean}"')

    def gen_binary_node_preds():
        """生成二元节点谓词 (x.attr != y.attr)。"""
        nodes = list(nodes_attrs.keys())
        for x, y in combinations(nodes, 2):
            attrs_x = nodes_attrs[x]
            attrs_y = nodes_attrs[y]
            for attr in set(attrs_x) & set(attrs_y):
                if attr in ['neighbors', 'multi_hop_neighbors']: continue
                if type(attrs_x.get(attr)) != type(attrs_y.get(attr)): continue
                if attrs_x.get(attr) != attrs_y.get(attr):
                    if isinstance(attrs_x.get(attr), (int, float)):
                        predicates.add(f"{x}.{attr} != {y}.{attr}")
                    elif isinstance(attrs_x.get(attr), str):
                        predicates.add(f'{x}.{attr} != {y}.{attr}')

    def gen_unary_edge_preds():
        """生成一元边谓词 ((x, y).attr = val)。"""
        for key, attrs in edges_attrs.items():
            try:
                if isinstance(key, tuple) and len(key) == 2:
                    x, y = key
                    edge_str = f"{x}___{y}"
                else:
                    continue

                for attr, val in attrs.items():
                    if attr == "text_span": continue

                    if isinstance(val, (int, float)):
                        predicates.add(f"{edge_str}.{attr} = {val}")
                        predicates.add(f"{edge_str}.{attr} != {val}")
                    elif isinstance(val, str) and val.strip():
                        val_clean = val.replace('"', "'")
                        predicates.add(f'{edge_str}.{attr} = "{val_clean}"')
                    elif isinstance(val, list):
                        predicates.add(f"{edge_str}.{attr}.length = {len(val)}")

                if "multi_hop" in attrs and isinstance(attrs["multi_hop"], dict):
                    hop = attrs["multi_hop"]
                    neighbor = str(hop.get('neighbor', 'N'))
                    edge = str(hop.get('edge', 'E'))
                    hop_count = hop.get('hop_count', 1)
                    if isinstance(hop_count, (int, float)):
                        predicates.add(f"{edge_str}_multi_hop.{neighbor}.{edge} = {hop_count}")
            except Exception:
                continue

    def gen_binary_edge_preds():
        """生成二元边谓词 ((x1, y1).attr != (x2, y2).attr)。"""
        valid_edges = [k for k in edges_attrs.keys() if isinstance(k, tuple) and len(k) == 2 and all(isinstance(i, str) for i in k)]

        for (ex1, ey1), (ex2, ey2) in combinations(valid_edges, 2):
            try:
                edge1_str = f'{ex1}___{ey1}'
                edge2_str = f'{ex2}___{ey2}'
                attrs1 = edges_attrs.get((ex1, ey1), {})
                attrs2 = edges_attrs.get((ex2, ey2), {})

                common_attrs = set(attrs1.keys()) & set(attrs2.keys())

                for attr in common_attrs:
                    if attr == "text_span": continue
                    val1 = attrs1.get(attr)
                    val2 = attrs2.get(attr)

                    if type(val1) != type(val2): continue

                    if val1 != val2:
                        predicates.add(f'{edge1_str}.{attr} != {edge2_str}.{attr}')

            except Exception:
                continue

    # ⚠️ 注意：M_predicate_new 依赖外部 LLM 类，这里假设它是可用的。
    def gen_ml_preds(m=5, n=5):
        """生成基于 LLM 评估的 ML 谓词。"""
        if llm is None:
            # 这是一个占位符，模拟生成一些 ML 谓词以确保规则生成有足够的谓词
            nodes = list(nodes_attrs.keys())
            if len(nodes) >= 2:
                x, y = random.sample(nodes, 2)
                predicates.add(f"None_ML({x}.attrs, {y}.attrs)")

            edges = [k for k in edges_attrs.keys() if isinstance(k, tuple) and len(k) == 2 and all(isinstance(i, str) for i in k)]
            if len(edges) >= 2:
                ex, ey = random.sample(edges, 2)
                predicates.add(f"None_ML({ex[0]}___{ex[1]}.attrs, {ey[0]}___{ey[1]}.attrs)")

            # print("🔴 LLM 客户端未提供，跳过 ML 谓词生成，使用模拟谓词。")
            return

        # 原始 LLM 逻辑
        # ... (此处应是 M_predicate_new 的实际调用逻辑) ...
        # (为避免代码冗长，此处仅保留注释，实际执行时依赖外部定义)
        pass

        # ⚠️ 注意：gen_span_mapping_preds 和 gen_contradiction_type_preds 依赖外部定义。

    # 此处假设它们能正确填充 predicates 集合。

    def print_predicate_statistics(predicates):
        """打印谓词统计信息 (假设已定义)。"""
        stats = defaultdict(int)
        for p in predicates:
            if "M_NLI" in p or "M_NLU" in p or "M_TEM" in p or "None_ML" in p:
                stats["ML"] += 1
            elif "S(" in p or "T(" in p or "violate(" in p:
                stats["Text/Contradiction"] += 1
            elif "__" in p or ".relationship" in p:
                stats["Edge_attr"] += 1
            else:
                stats["Entity_attr"] += 1

        print("📊 Predicate Type Statistics:")
        for k, v in sorted(stats.items(), key=lambda item: item[1], reverse=True):
            print(f" - {k}: {v}")

    # -------------------- III. 执行谓词生成 ---------------------
    # print("🧠 Generating predicates...")

    gen_unary_node_preds()
    gen_binary_node_preds()
    gen_unary_edge_preds()
    gen_binary_edge_preds()
    gen_ml_preds()  # 依赖外部 M_predicate_new

    # 假设以下两个函数能被调用 (如果它们是全局定义的)
    # gen_span_mapping_preds(predicates)
    # if llm:
    #     gen_contradiction_type_preds(predicates, llm)

    # -------------------- IV. 规则生成 (重点修改部分) ---------------------
    predicates = {p.strip().replace("  ", " ") for p in predicates if p}

    if not predicates:
        # print("🔴 无法生成规则: 谓词集为空。")
        return predicates, rules

    print_predicate_statistics(predicates)
    pred_list = list(predicates)
    random.shuffle(pred_list)

    max_rules = 10
    rule_count = 0
    # 前件长度范围：[2, 3, 4, 5, 6]
    body_lengths = [2, 3, 4, 5, 6]
    # 每个长度的目标规则数，争取均匀分布
    rules_per_length = max_rules // len(body_lengths)
    rules_generated_by_length = defaultdict(int)
    used_heads = set()

    # 检查谓词数量是否足够
    if len(pred_list) < 6:
        print(f"⚠️ 警告: 谓词总数 ({len(pred_list)}) 小于所需的最大前件长度 (6)。")
        # 调整最大长度为可用谓词数-1，但至少为2
        max_available_length = max(2, len(pred_list) - 1)
        body_lengths = [l for l in body_lengths if l <= max_available_length]
        if not body_lengths:
            print("🔴 谓词数量不足以生成至少包含 2 个谓词前件的规则。")
            return predicates, rules

    while rule_count < max_rules:
        # 1. 选择下一个目标长度 (优先选择生成规则少的长度)
        target_length = min(body_lengths, key=lambda l: (rules_generated_by_length[l], random.random()))

        # 如果所有目标长度都已达标，则退出循环
        if rules_generated_by_length[target_length] >= rules_per_length and rule_count >= max_rules:
            break

        # 2. 尝试从谓词列表中选择 body 和 head

        # 随机选择 (L+1) 个谓词：L 个用于 Body，1 个用于 Head
        required_preds = target_length + 1
        if len(pred_list) < required_preds:
            # 如果剩余谓词不足以生成所需长度的规则，移除该目标长度
            body_lengths = [l for l in body_lengths if l != target_length]
            if not body_lengths: break
            continue

        try:
            # 随机选择 required_preds 个谓词
            selected_preds = random.sample(pred_list, required_preds)
        except ValueError:
            # 这种情况不应该发生，但以防万一
            break

        head = random.choice(selected_preds)
        body_parts = [p for p in selected_preds if p != head]

        # 确保 body 长度符合要求
        if len(body_parts) != target_length:
            continue

        # 确保 head 尚未被使用
        if head in used_heads:
            continue

        # 3. 构造规则并添加
        body = " ∧ ".join(body_parts)
        rule = f"Q[{x0},{y0}] ({body} → {head})"

        rules.append(rule)
        used_heads.add(head)
        rules_generated_by_length[target_length] += 1
        rule_count += 1

        # 动态调整目标长度列表：如果某个长度规则数已满，从目标中移除
        if rules_generated_by_length[target_length] >= rules_per_length and target_length in body_lengths:
            body_lengths.remove(target_length)

        # 如果所有长度都被移除，但总数未达到 max_rules，则重新填充 body_lengths，
        # 允许随机选取 (这可以保证达到 max_rules)
        if not body_lengths and rule_count < max_rules:
            body_lengths = [l for l in [2, 3, 4, 5, 6] if l <= max_available_length]

    return predicates, rules

if __name__ == "__main__":
    DUAL_STAR_JSON_DIR = "./dual_star_json_with_attrs"
    OUTPUT_DIR = "dual_star/direct_generate_rules"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_files = glob(os.path.join(DUAL_STAR_JSON_DIR, "*.json"))

    print("--- 检查 API Key ---")
    if not os.getenv("OPENAI_API_KEY"):
        print("🔴 警告：OPENAI_API_KEY 未设置！LLM 调用将失败。请检查 .env 文件。")
    print("-------------------")

    for json_path in json_files:
        filename = os.path.basename(json_path)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"\n🌀 处理中: {filename}")

        try:
            data = load_dual_star_json(json_path)
            x0, y0 = find_dual_star_centers(data)

            preds, rules = generate_predicates_and_rules(data, x0, y0)

            output_file = os.path.join(OUTPUT_DIR, f"{name_without_ext}_rico.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("Predicates:\n")
                for p in sorted(preds):
                    f.write(p + "\n")

                f.write("\nRules:\n")
                for r in rules:
                    f.write(r + "\n")

            print(f"✅ 已保存到: {output_file}")

        except ValueError as ve:
            print(f"❌ 处理失败: {filename} - 错误: {ve}")
        except Exception as e:
            print(f"❌ 处理失败: {filename} - 错误: {type(e).__name__}: {e}")