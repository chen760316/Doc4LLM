"""
用 Tree-LSTM 替换 PCFG 生成规则的思路

训练阶段：
使用已有的规则（树结构）作为训练样本。
用 Tree-LSTM 编码规则的树结构，学习节点之间的组合关系和语义。
训练一个解码器（通常也是基于 Tree-LSTM 或 RNN）来生成新的规则树。

生成阶段：
给定初始条件（如起始非终结符或部分谓词），用训练好的 Tree-LSTM 解码器生成新的规则树。
通过采样策略（beam search、贪婪等）生成符合训练数据分布的新规则。

实现步骤建议
（1）准备训练数据（树结构化规则）
规则需要解析成树结构（例如，每条规则解析成抽象语法树AST）。
例如：Q[x0,y0] (p1 ∧ p2 → p3) 解析成根节点Q，左子树是前提集合，右子树是结论。

（2）定义 Tree-LSTM 编码器
参考论文：Tai et al., 2015的 Child-Sum Tree-LSTM
输入：节点特征（谓词或符号的embedding）。
输出：根节点的隐状态向量，表示整个规则。

（3）定义 Tree-LSTM 解码器
从根开始递归生成子节点，直到叶节点。
利用隐状态预测下一层节点的符号/谓词。
结合采样策略产生多条规则。

（4）训练过程
最小化解码器生成树与真实树结构之间的损失（交叉熵或结构化损失）。
用已知规则树做监督。

（5）生成规则
输入初始条件或随机向量，解码器递归生成规则树。
将树结构转换为文本规则格式。

具体说明:
1.训练阶段:
先用 load predicates and_rules 加载已有的谓词(predicates)和规则(rules)。
将规则文本用 parse_ruletotree解析成树结构(RuleTreeNode)
通过 TreeLsTMEncoder 递归编码每棵规则树，得到树的隐藏表示(隐状态向量)。
TreeLsTMDecoder 根据编码器输出的隐藏状态，学习解码生成树结构(递归生成节点符号和结构)
用交叉熵损失(nn.crossEntropyLoss)训练编码器+解码器，使其能重构训练数据的规则树结构
2生成阶段:
训练好后，给解码器一个初始隐状态(h_init ，此处是零向量)，解码器递归生成规则树(调用decode 函数)。
生成的规则树可转换成文本规则(treetorule_str)。
生成多条规则样例并打印，同时保存到文件。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
from typing import List, Tuple, Optional
from itertools import combinations

import re

import os
# 设定你的目录
OUTPUT_DIR = "predicate_rule_generation/lstm"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def classify_predicate(pred: str) -> str:
    pred = pred.strip()
    if pred.startswith("linkage"):
        return "linkage"
    elif pred.startswith("M_NLI") or pred.startswith("M_NLU") or pred.startswith("M_TEM"):
        return "ml_pred"
    elif pred.startswith("contra"):
        return "contra"
    elif re.match(r'.+_.+\..+\s*[=!<>]+', pred):  # x_y.attr op val
        return "edge_attr"
    elif re.match(r'.+\..+\s*[=!<>]+', pred):
        return "node_attr"
    elif re.match(r'.+_.+\.length\s*=', pred):
        return "edge_attr_list"
    elif re.match(r'.+\..+\s*!=\s*.+\..+', pred):
        return "node_attr_binary"
    else:
        return "unknown"


# —— 数据加载函数 —— #
from glob import glob
import json
import os
import random
import re
from collections import defaultdict
from itertools import combinations
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from collections import Counter
import random

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 初始化LLM客户端
llm = OpenAIClient(
    model="gpt-4.1-mini",
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


    return list(predicates), rules

def find_dual_star_centers(data):
    """
    在双星结构中，寻找中心节点（通过 same_entity 关系连接的两个节点）。
    返回中心节点名 x0, y0。
    """
    for triple in data:
        rel = triple.get("relationship", "").lower()
        if rel == "same_entity":
            node1 = triple.get("node_1", {}).get("name")
            node2 = triple.get("node_2", {}).get("name")
            if node1 and node2:
                return node1, node2

    raise ValueError("❌ 未找到 same_entity 的中心节点连接")

# —— 规则树结构定义 —— #
class RuleTreeNode:
    def __init__(self, symbol: str, children: Optional[List['RuleTreeNode']] = None):
        self.symbol = symbol
        self.children = children if children else []

    def __repr__(self):
        if not self.children:
            return self.symbol
        return f"{self.symbol}({', '.join(repr(c) for c in self.children)})"

# —— 词表 —— #
class Vocab:
    def __init__(self, tokens: List[str]):
        self.token2idx = {tok: i for i, tok in enumerate(tokens)}
        self.idx2token = {i: tok for i, tok in enumerate(tokens)}
        self.unk = "<UNK>"
        if self.unk not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[self.unk] = idx
            self.idx2token[idx] = self.unk

    def encode(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[self.unk])

    def decode(self, idx: int) -> str:
        return self.idx2token.get(idx, self.unk)

    def __len__(self):
        return len(self.token2idx)

# —— TreeLSTM 单元 —— #
class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super().__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(in_dim, 3 * mem_dim)
        self.iouh = nn.Linear(mem_dim, 3 * mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)

    def forward(self, inputs, child_c, child_h):
        # inputs: (batch, in_dim) here batch usually 1
        if len(child_c) == 0:
            child_c_sum = torch.zeros(inputs.size(0), self.mem_dim, device=inputs.device)
            child_h_sum = torch.zeros(inputs.size(0), self.mem_dim, device=inputs.device)
        else:
            child_c_sum = torch.stack(child_c, dim=0).sum(0)
            child_h_sum = torch.stack(child_h, dim=0).sum(0)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        f_terms = []
        for h in child_h:
            f_terms.append(torch.sigmoid(self.fh(h)))
        if len(f_terms) == 0:
            fc = 0
        else:
            f_stack = torch.stack(f_terms, dim=0)
            c_stack = torch.stack(child_c, dim=0)
            fc = (f_stack * c_stack).sum(0)

        c = i * u + fc
        h = o * torch.tanh(c)
        return c, h

# —— 编码器 —— #
class TreeLSTMEncoder(nn.Module):
    def __init__(self, vocab: Vocab, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), embed_dim)
        self.cell = ChildSumTreeLSTMCell(embed_dim, hidden_dim)

    def forward(self, node: RuleTreeNode) -> Tuple[torch.Tensor, torch.Tensor]:
        # 递归编码
        # 把 symbol 编码为索引，再 embedding
        idx = self.vocab.encode(node.symbol)
        embed = self.embedding(torch.tensor([idx]))
        child_states = [ self.forward(child) for child in node.children ]
        child_c = [c for c, h in child_states]
        child_h = [h for c, h in child_states]

        c, h = self.cell(embed, child_c, child_h)
        return c, h

# —— 解码器 —— #
class TreeLSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, vocab: Vocab, max_children: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cell = ChildSumTreeLSTMCell(embed_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.max_children = max_children

        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.child_num_predictor = nn.Linear(hidden_dim, max_children + 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.output_layer(h)
        child_num_logits = self.child_num_predictor(h)
        return logits, child_num_logits

    def decode(self, h: torch.Tensor, depth: int = 0, max_depth: int = 4) -> Optional[RuleTreeNode]:
        if depth > max_depth:
            return None

        # 固定根节点为 Q
        root = RuleTreeNode("Q")

        # 生成前提和结论子树
        # —— PREMISE —— #
        premise_node = RuleTreeNode("PREMISE")
        num_premises = random.randint(1, 3)  # 可改为 child_num_predictor

        for _ in range(num_premises):
            # 随机选一个 predicate
            pred_idx = random.randint(0, self.vocab_size - 1)
            pred_symbol = self.vocab.decode(pred_idx)
            predicate_node = RuleTreeNode(pred_symbol)
            premise_node.children.append(predicate_node)

        # —— CONCLUSION —— #
        # 再生成结论谓词
        concl_idx = random.randint(0, self.vocab_size - 1)
        concl_symbol = self.vocab.decode(concl_idx)
        conclusion_node = RuleTreeNode(concl_symbol)

        root.children = [premise_node, conclusion_node]
        return root


# —— 规则解析函数，把规则字符串解析为树结构 —— #
def parse_rule_to_tree(rule_str: str) -> RuleTreeNode:
    """
    假设规则形如：
      Q[x0,y0] (p1 ∧ p2 → p3)
    """
    m = re.match(r'Q\[[^\]]+\]\s*\((.*)→(.*)\)', rule_str)
    if not m:
        # 如果格式不匹配，可以返回一个简单节点或者 raise
        return RuleTreeNode(rule_str.strip())
    premise = m.group(1).strip()
    conclusion = m.group(2).strip()
    # 分割前提部分
    premise_preds = [p.strip() for p in premise.split("∧") if p.strip()]
    premise_node = RuleTreeNode("PREMISE", [RuleTreeNode(p) for p in premise_preds])
    root = RuleTreeNode("Q", [premise_node, RuleTreeNode(conclusion)])
    return root

# —— 训练 epoch —— #
def train_epoch(encoder, decoder, train_trees: List[RuleTreeNode],
                optimizer, criterion_symbol, criterion_childnum, vocab: Vocab):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for tree in train_trees:
        optimizer.zero_grad()
        c, h = encoder(tree)  # encode
        # 解码部分，用根 h 作预测节点symbol 和子节点数量
        logits, child_num_logits = decoder.forward(h)
        # 标签
        target_symbol = torch.tensor([vocab.encode(tree.symbol)])
        target_child_num = torch.tensor([len(tree.children)])
        loss_symbol = criterion_symbol(logits, target_symbol)
        loss_childnum = criterion_childnum(child_num_logits, target_child_num)
        loss = loss_symbol + loss_childnum
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_trees)

# —— 工具：树转文本 —— #
def tree_to_rule_str(node: RuleTreeNode, vars: str = "x0,y0") -> str:
    if node is None:
        return ""
    if not node.children:
        return node.symbol
    if node.symbol == "Q" and len(node.children) == 2 and node.children[0].symbol == "PREMISE":
        premise_preds = [tree_to_rule_str(c) for c in node.children[0].children]
        conclusion = tree_to_rule_str(node.children[1])
        return f"Q[{vars}] ({' ∧ '.join(premise_preds)} → {conclusion})"
    return node.symbol + "(" + ", ".join(tree_to_rule_str(c) for c in node.children) + ")"

# 遍历新规则树的所有谓词节点，确保它们格式正确
def validate_tree(tree: RuleTreeNode) -> bool:
    if tree is None:
        return True
    # 用 classify_predicate 判断节点符号
    pred_type = classify_predicate(tree.symbol)
    if pred_type == "unknown":
        print(f"发现非法谓词: {tree.symbol}")
        return False
    # 递归检查所有子节点
    return all(validate_tree(child) for child in tree.children)


# —— 主流程 —— #
def main():
    # 遍历所有 JSON 文件
    DUAL_STAR_JSON_DIR = "../star_json/dual_graph"
    json_files = glob(os.path.join(DUAL_STAR_JSON_DIR, "dual_star_graph_1_Emily_Clark.json"))

    # —— 步骤 2: 加载图数据（用于生成新谓词） ——
    with open("../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # 加载 JSON 数据
    # 提取文件路径
    data = load_dual_star_json(json_files[0])
    x0, y0 = find_dual_star_centers(data)

    # 推测对应文本路径，比如 text1/2 位于 texts 文件夹中
    # 你需要保证这些文件存在或按规则命名，例如：
    # dual_star_graph_2_John_Smith.json -> text1_dual_star_graph_2_John_Smith.txt
    text1_path = "text1.txt"
    text2_path = "text2.txt"

    if not os.path.exists(text1_path) or not os.path.exists(text2_path):
        print(f"⚠️ 缺失对应文本文件: {text1_path} 或 {text2_path}")

    doc1 = load_text_file(text1_path)
    doc2 = load_text_file(text2_path)

    preds, rules = generate_predicates_and_rules(data, doc1, doc2, x0, y0)
    print("📊 新生成的谓词数量:", len(preds))

    # 2. 基于规则文本生成训练树
    train_trees = []
    for r in rules:
        tree = parse_rule_to_tree(r)
        train_trees.append(tree)

    # 3. 构造词表：包括规则里的所有符号 + special ones
    #    也可以加上你 train_preds 中的 predicate 表示为符号
    symbols = set()
    for tree in train_trees:
        def dfs(n: RuleTreeNode):
            symbols.add(n.symbol)
            for ch in n.children:
                dfs(ch)
        dfs(tree)
    # 可加 “PREMISE”,"Q" 等 if not already
    # 如果你的 predicates 有额外节点，也可以加
    symbols = list(symbols)
    vocab = Vocab(symbols)

    # 4. 初始化模型
    EMBED_DIM = 128
    HIDDEN_DIM = 128
    LR = 0.01
    EPOCHS = 50

    encoder = TreeLSTMEncoder(vocab, EMBED_DIM, HIDDEN_DIM)
    decoder = TreeLSTMDecoder(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, vocab=vocab, max_children=3)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    criterion_symbol = nn.CrossEntropyLoss()
    criterion_childnum = nn.CrossEntropyLoss()

    # 5. 训练
    for epoch in range(EPOCHS):
        loss = train_epoch(encoder, decoder, train_trees, optimizer, criterion_symbol, criterion_childnum, vocab)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss:.4f}")

    # 6. 生成新规则
    encoder.eval()
    decoder.eval()

    # 设置输出文件路径
    output_file_path = os.path.join(OUTPUT_DIR, "generated_rules_lstm.txt")

    print(f"文件已保存到: {output_file_path}")

    with open(output_file_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            h_init = torch.zeros(1, HIDDEN_DIM)
            num_rules = 5  # 你想生成的规则数
            for i in range(num_rules):
                new_tree = decoder.decode(h_init, max_depth=3)
                print("*" * 100)
                print(f"\n生成的新规则树 #{i + 1}：")
                print(new_tree)
                print("对应规则文本：")
                print(tree_to_rule_str(new_tree))
                f.write(tree_to_rule_str(new_tree) + "\n")

if __name__ == "__main__":
    main()

