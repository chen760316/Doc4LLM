"""
根据（3）得到的双星结构，从中生成示例的RICO谓词和规则，提供给生成式模型进行学习
候选谓词w为 violate(𝑥0,𝑦0) ，该谓词在每个双星模式上只有一个，取值为八种具体矛盾类型之一，目标在指出该双星的具体矛盾类型
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

def compute_conflict_score_llm(premise, hypothesis):
    system_message = "You are a natural language inference assistant."
    user_message = f"""
    Determine whether the following two sentences contradict each other.

    Please return a single decimal number between 0 and 1 representing the probability of contradiction,
    where 0 means no contradiction and 1 means strong contradiction.

    Premise: "{premise}"
    Hypothesis: "{hypothesis}"

    Only return the decimal number.
    """

    response = llm.generate(system_message, user_message)
    content = response.strip()  # Assuming response is a string

    try:
        score = float(content)
        score = max(0.0, min(1.0, score))
    except Exception as e:
        print("Failed to parse score, returning 0.0 by default.", e)
        score = 0.0

    return score


# Model 2: Contextual Dependency and Co-occurrence
def judge_dependency_and_cooccurrence(sentence_a: str, sentence_b: str) -> tuple[float, float, str]:
    system_message = (
        "You are an assistant that evaluates two sentences for:\n"
        "1) Contextual dependency — how much the meaning of one sentence depends on the other.\n"
        "2) Co-occurrence — how often these two sentences/events appear together in the same context.\n"
        "Return two decimal numbers between 0 and 1 separated by a comma, "
        "first is dependency score, second is co-occurrence score."
    )

    user_message = f"""
    Sentence A: "{sentence_a}"
    Sentence B: "{sentence_b}"

    Please respond ONLY with two decimal numbers separated by a comma, e.g. "0.45, 0.78"
    """

    response = llm.generate(system_message, user_message)
    dep_str, cooc_str = response.strip().split(",")
    dep_score = float(dep_str.strip())
    cooc_score = float(cooc_str.strip())

    dep_score = max(0.0, min(1.0, dep_score))
    cooc_score = max(0.0, min(1.0, cooc_score))

    dependency_threshold = 0.8
    cooccurrence_threshold = 0.6

    if dep_score >= dependency_threshold:
        decision = "dependency"
    elif cooc_score >= cooccurrence_threshold:
        decision = "co-occurrence"
    else:
        decision = "neither"

    return dep_score, cooc_score, decision


# Model 3: Temporal Inference
def infer_temporal_relation_openai(event_a: str, event_b: str) -> tuple[str, float]:
    system_message = (
        "You are an assistant that determines the temporal relationship between two events. "
        "Your task is to answer whether event A happened before event B. "
        "Answer with only one word: 'yes', 'no', or 'unknown'."
    )

    user_message = f"""
    Given two events:
    A: {event_a}
    B: {event_b}

    Question: Did event A happen before event B?
    Answer with only one word: yes, no, or unknown.
    """

    response = llm.generate(system_message, user_message)
    answer = response.strip().lower()
    score_map = {
        "yes": 1.0,
        "no": 0.0,
        "unknown": 0.5
    }
    score = score_map.get(answer, 0.5)

    return answer, score


# Combine models into one function
def M_predicate_new(attrs1: dict, attrs2: dict, ml_threshold: float = 0.5) -> str:
    """
    Given two sets of attributes, calculate the highest score from three models:
    1) Conflict score (contradiction/entailment)
    2) Contextual dependency and co-occurrence
    3) Temporal inference

    Returns a label indicating the highest probability.
    """
    # Extract relevant textual information from the attributes
    premise = attrs1.get("sentence", "")
    hypothesis = attrs2.get("sentence", "")

    # Model 1: Conflict Score
    conflict_score = compute_conflict_score_llm(premise, hypothesis)

    # Model 2: Contextual Dependency and Co-occurrence
    dep_score, cooc_score, decision = judge_dependency_and_cooccurrence(premise, hypothesis)

    # Model 3: Temporal Inference
    ans, temporal_score = infer_temporal_relation_openai(premise, hypothesis)

    # Choose the highest score
    scores = {
        "conflict": conflict_score,
        "dependency": dep_score,
        "co-occurrence": cooc_score,
        "temporal": temporal_score
    }

    highest_score_model = max(scores, key=scores.get)
    highest_score = scores[highest_score_model]

    # Find dynamic identifiers based on available keys in attrs1 and attrs2
    # Priority 1: Use 'name' if available
    # Priority 2: Use the first available key from the dictionary (if 'name' is missing)
    def get_identifier(attrs: dict):
        # Take the first key-value pair if no name exists
        return f"({', '.join([f'{k}: {v}' for k, v in attrs.items()])})"

    # 只取一个属性
    # def get_identifier(attrs: dict):
    #     # Check if 'name' field exists
    #     if "name" in attrs:
    #         return f"name: {attrs['name']}"
    #     else:
    #         # If 'name' doesn't exist, return the first key-value pair
    #         first_key, first_value = next(iter(attrs.items()))
    #         return f"{first_key}: {first_value}"
    #

    identifier_1 = get_identifier(attrs1)
    identifier_2 = get_identifier(attrs2)

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
- If they do, classify the contradiction into one of: ["Factual", "Content", "Causal", "Negation", "Relation", "Numeric", "Temporal"].
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

def contra_from_spans(span_x, span_y, node_x, node_y, doc1, doc2):
    """
    提取文本片段并返回 contra("文本1", "文本2") 谓词
    """
    text1 = doc1[span_x["start"]:span_x["end"]].strip().replace('"', "'")
    text2 = doc2[span_y["start"]:span_y["end"]].strip().replace('"', "'")
    return f'contra("{text1}", "{text2}")'


def contra(s1: str, s2: str, llm) -> bool:
    """
    Returns True if s1 and s2 contradict each other (semantically), using LLM.
    """
    system_prompt = (
        "You are a natural language inference engine. "
        "Determine whether the following two sentences contradict each other. "
        "Respond with only one word: 'Yes' if they contradict, or 'No' if they do not."
    )

    user_message = f"Sentence 1: {s1}\nSentence 2: {s2}\nDo they contradict?"

    try:
        response = llm.generate(
            user_message=user_message,
            system_message=system_prompt
        )
        answer = response.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"⚠️ Error in contra() check:\n{e}")
        return False  # fallback

# ML predicate伪函数
def M_predicate(attrs1: dict, attrs2: dict) -> float:
    common_keys = set(attrs1.keys()) & set(attrs2.keys())
    score = sum(1 for k in common_keys if attrs1[k] == attrs2[k]) / max(len(common_keys), 1)
    return score

def load_dual_star_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_document_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def generate_predicates_and_rules(data, doc1, doc2, x0, y0, ml_threshold=0.5):
    predicates = set()
    rules = []

    nodes_attrs = {}
    edges_attrs = {}
    text_spans = {}

    for triple in data:
        n1 = triple["node_1"]["name"]
        n2 = triple["node_2"]["name"]

        for k, v in triple["node_1"].items():
            if k not in ["name", "text_span", "label"]:
                nodes_attrs.setdefault(n1, {})[k] = v
        for k, v in triple["node_2"].items():
            if k not in ["name", "text_span", "label"]:
                nodes_attrs.setdefault(n2, {})[k] = v

        if "text_span" in triple["node_1"]:
            text_spans[n1] = triple["node_1"]["text_span"]
        if "text_span" in triple["node_2"]:
            text_spans[n2] = triple["node_2"]["text_span"]

        # 边属性：先初始化字典
        edge_key = (n1, n2)
        edges_attrs.setdefault(edge_key, {})

        # 关系字符串
        if "relationship" in triple and triple["relationship"]:
            edges_attrs[edge_key]["relationship"] = triple["relationship"]

        # 加入edge_properties中的所有属性
        if "edge_properties" in triple:
            for ek, ev in triple["edge_properties"].items():
                edges_attrs[edge_key][ek] = ev

    def gen_unary_preds():
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
        for (x, y), attrs in edges_attrs.items():
            for attr, val in attrs.items():
                if attr == "text_span":
                    continue  # 跳过文本span字段
                if isinstance(val, (int, float)):
                    predicates.add(f"{x}_{y}.{attr} = {val}")
                    predicates.add(f"{x}_{y}.{attr} != {val}")
                    predicates.add(f"{x}_{y}.{attr} < {val + 1}")
                    predicates.add(f"{x}_{y}.{attr} > {val - 1}")
                    predicates.add(f"{x}_{y}.{attr} <= {val}")
                    predicates.add(f"{x}_{y}.{attr} >= {val}")
                elif isinstance(val, str):
                    predicates.add(f'{x}_{y}.{attr} = "{val}"')
                    predicates.add(f'{x}_{y}.{attr} != "{val}"')
                elif isinstance(val, list):
                    # 简单处理list，转成字符串摘要或长度谓词
                    predicates.add(f"{x}_{y}.{attr}.length = {len(val)}")

    def gen_binary_edge_preds():
        edges = list(edges_attrs.keys())
        for (ex1, ey1), (ex2, ey2) in combinations(edges, 2):
            attr1 = edges_attrs[(ex1, ey1)]["relationship"]
            attr2 = edges_attrs[(ex2, ey2)]["relationship"]
            if attr1 != attr2:
                predicates.add(f'{ex1}_{ey1}.relationship != {ex2}_{ey2}.relationship')

    def gen_contra_preds():
        nodes = list(text_spans.keys())
        for x, y in combinations(nodes, 2):
            span_x = text_spans[x]
            span_y = text_spans[y]

            # 判断两个节点属于哪个文档
            # 假设 node_1 来源于 text1，node_2 来源于 text2
            doc_x = doc1 if "doc1" in x.lower() or "node_1" in x.lower() else doc1  # 默认都来自 doc1
            doc_y = doc2 if "doc2" in y.lower() or "node_2" in y.lower() else doc2  # 可根据实际命名规则改

            contra_pred = contra_from_spans(span_x, span_y, x, y, doc_x, doc_y)
            predicates.add(contra_pred)

    def print_predicate_statistics(predicates):
        stats = defaultdict(int)
        for p in predicates:
            if "contra" in p:
                stats["contra"] += 1
            elif "M_NLI" in p or "M_NLU" in p or "M_TEM" in p:
                stats["ML"] += 1
            elif "relationship" in p or "_.relationship" in p:
                stats["edge_attr"] += 1
            else:
                stats["entity_attr"] += 1

        print("📊 Predicate Type Statistics:")
        for k, v in stats.items():
            print(f" - {k}: {v}")


    def gen_ml_preds():
        nodes = list(nodes_attrs.keys())
        for x, y in combinations(nodes, 2):
            score = M_predicate_new(nodes_attrs[x], nodes_attrs[y])
            predicates.add(score)

    # 执行所有谓词生成器
    predicates = set()
    gen_unary_preds()
    gen_binary_node_preds()
    gen_unary_edge_preds()
    gen_binary_edge_preds()
    gen_contra_preds()
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
    max_rules = 20

    # 先按类型分类谓词
    type_buckets = defaultdict(list)
    for p in pred_list:
        if "contra" in p:
            type_buckets["contra"].append(p)
        elif "M(" in p:
            type_buckets["M"].append(p)
        else:
            type_buckets["other"].append(p)

    # 先确保每个类型至少出现一次作为后件
    used_heads = set()
    for t, preds_of_type in type_buckets.items():
        # 先随便挑一个谓词作为后件
        head = random.choice(preds_of_type)
        used_heads.add(head)
        # 找个body（不包含head）组合成规则
        bodies = [p for p in pred_list if p != head]
        if not bodies:
            continue
        body = random.choice(bodies)
        rule_body = f"{body} ∧ {head}"
        rules.append(f"Q[{x0},{y0}] ({rule_body} → {head})")
        rule_count += 1
        if rule_count >= max_rules:
            break

    if rule_count < max_rules:
        # 生成剩余规则，随机后件，但不重复用已经用过的head，保证多样性
        combos = list(combinations(pred_list, 2))
        random.shuffle(combos)
        for combo in combos:
            body = " ∧ ".join(combo)
            # 随机选后件
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
    在双星结构中，寻找中心节点（通过 same_entity 关系连接的两个节点）。
    返回中心节点名 x0, y0。
    """
    for triple in data:
        rel = triple.get("relationship", "").lower()
        if rel == "semantically_related":
            node1 = triple.get("node_1", {}).get("name")
            node2 = triple.get("node_2", {}).get("name")
            if node1 and node2:
                return node1, node2

    raise ValueError("❌ 未找到 same_entity 的中心节点连接")

if __name__ == "__main__":
    # 设定你的目录
    DUAL_STAR_JSON_DIR = "../semantic_dual_star_json"
    OUTPUT_DIR = "semantic_dual_star"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 遍历所有 JSON 文件
    json_files = glob(os.path.join(DUAL_STAR_JSON_DIR, "*.json"))
    # 只取前两个json
    json_files = json_files[:2]

    for json_path in json_files:
        filename = os.path.basename(json_path)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"\n🌀 处理中: {filename}")

        try:
            # 加载 JSON 数据
            data = load_dual_star_json(json_path)
            x0, y0 = find_dual_star_centers(data)

            # 推测对应文本路径，比如 text1/2 位于 texts 文件夹中
            # 你需要保证这些文件存在或按规则命名，例如：
            # dual_star_graph_2_John_Smith.json -> text1_dual_star_graph_2_John_Smith.txt
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
