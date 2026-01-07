"""
PCFG 的思路是把规则结构看成一种语法树／语言，用非终结符（non‑terminals）和终结符（terminators）定义生成规则的语法，然后训练这个 PCFG，以概率的方式生成新的规则。

✅ 核心步骤
定义语法：定义规则句法结构（例如 “RULE → Q [ x0 , y0 ] ( PREMISE → CONCLUSION )”／“PREMISE → PREDICATE ∧ PREMISE | PREDICATE”／“CONCLUSION → PREDICATE”…）。
从训练规则中统计语法产生式频率：从已有规则中抽出树结构／语法分解，统计产生式频率，用这些频率决定概率。
采样生成新规则：按照 PCFG 的概率采样生成新的规则。
"""

import random
import re
import json
from collections import Counter, defaultdict
from typing import List, Tuple
from itertools import combinations


# ——— 谓词类型分类函数 ———
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

# ——— 加载谓词和规则 ———
def load_predicates_and_rules(file_path: str) -> Tuple[List[str], List[str]]:
    predicates = []
    rules = []
    mode = None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == "Predicates:":
                mode = "pred"
                continue
            elif line == "Rules:":
                mode = "rules"
                continue
            if not line or mode is None:
                continue
            if mode == "pred":
                predicates.append(line)
            elif mode == "rules":
                rules.append(line)
    return predicates, rules

# ——— 你这个谓词 + 规则生成器的函数 ———
# 假设 data 是 triples 列表格式，doc1/doc2 文档标识或内容，不一定需要很复杂
def generate_predicates_and_rules(data, doc1, doc2, x0, y0, ml_threshold=0.5, max_rules=10) -> Tuple[List[str], List[str]]:
    predicates = set()
    rules = []

    nodes_attrs = {}
    edges_attrs = {}
    text_spans = {}

    # 提取节点属性、边属性、text span
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

        edge_key = (n1, n2)
        edges_attrs.setdefault(edge_key, {})

        if "relationship" in triple and triple["relationship"]:
            edges_attrs[edge_key]["relationship"] = triple["relationship"]

        if "edge_properties" in triple:
            for ek, ev in triple["edge_properties"].items():
                edges_attrs[edge_key][ek] = ev

    # 生成 predicate 类型
    def gen_unary_preds():
        for node, attrs in nodes_attrs.items():
            for attr, val in attrs.items():
                if isinstance(val, (int, float)):
                    predicates.add(f"{node}.{attr} = {val}")
                    predicates.add(f"{node}.{attr} != {val}")
                    # 数值比较可以加偏移
                    predicates.add(f"{node}.{attr} < {val + 1}")
                    predicates.add(f"{node}.{attr} > {val - 1}")
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
            shared = set(attrs_x.keys()) & set(attrs_y.keys())
            for attr in shared:
                # 类型要相同
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
                    continue
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
                    predicates.add(f"{x}_{y}.{attr}.length = {len(val)}")

    def gen_binary_edge_preds():
        edges = list(edges_attrs.keys())
        for (ex1, ey1), (ex2, ey2) in combinations(edges, 2):
            # 如果两个 relationship 存在并且不同
            rel1 = edges_attrs[(ex1, ey1)].get("relationship")
            rel2 = edges_attrs[(ex2, ey2)].get("relationship")
            if rel1 and rel2 and rel1 != rel2:
                predicates.add(f'{ex1}_{ey1}.relationship != {ex2}_{ey2}.relationship')

    def gen_contra_preds():
        # 简单做两节点 text_span 对比
        for x, y in combinations(text_spans.keys(), 2):
            span_x = text_spans[x]
            span_y = text_spans[y]
            # 假设 doc1/doc2 是标识 doc 来源
            # 这里只示例，不一定语义正确
            contra_pred = f'contra("{span_x}", "{span_y}")'
            predicates.add(contra_pred)

    def gen_ml_preds():
        # 示例：如果某节点属性向量相近或不同，就加 ML 型谓词
        for x, y in combinations(nodes_attrs.keys(), 2):
            # 这里 ml_score 模拟
            # 你应当替换成你真正的 ML 模型计算函数
            attrs_x = nodes_attrs[x]
            attrs_y = nodes_attrs[y]
            # 简单 heuristic：如果属性重叠少，则认为 score 小
            overlap = set(attrs_x.items()) & set(attrs_y.items())
            total = max(len(attrs_x), len(attrs_y), 1)
            score = len(overlap) / total
            if score < ml_threshold:
                predicates.add(f'M_NLI({x}.attrs, {y}.attrs)')
                predicates.add(f'M_NLU({x}.attrs, {y}.attrs)')
                predicates.add(f'M_TEM({x}.attrs, {y}.attrs)')

    # 执行生成
    gen_unary_preds()
    gen_binary_node_preds()
    gen_unary_edge_preds()
    gen_binary_edge_preds()
    gen_contra_preds()
    gen_ml_preds()

    # 清理谓词
    pred_list = list({p for p in predicates if p and isinstance(p, str)})

    # 输出 statistics
    stats = defaultdict(int)
    for p in pred_list:
        t = classify_predicate(p)
        stats[t] += 1
    print("📊 Predicate Type Statistics:")
    for t, cnt in stats.items():
        print(f" - {t}: {cnt}")

    # 生成规则
    rules_out = []
    used_heads = set()

    # 第一阶段：让每种类型至少一个 head
    type_buckets = defaultdict(list)
    for p in pred_list:
        t = classify_predicate(p)
        type_buckets[t].append(p)

    for t, bucket in type_buckets.items():
        if not bucket:
            continue
        head = random.choice(bucket)
        # 选择 body（至少一个谓词，不是 head）
        body_cands = [p for p in pred_list if p != head]
        if not body_cands:
            continue
        body = random.choice(body_cands)
        rule = f"Q[{x0},{y0}] ({body} ∧ {head} → {head})"
        rules_out.append(rule)
        used_heads.add(head)
        if len(rules_out) >= max_rules:
            break

    # 第二阶段：剩余规则
    all_pairs = list(combinations(pred_list, 2))
    random.shuffle(all_pairs)
    for (p1, p2) in all_pairs:
        if len(rules_out) >= max_rules:
            break
        head = p2
        if head in used_heads:
            continue
        # body 用 p1 ∧ p2 或两个不同类型
        rule = f"Q[{x0},{y0}] ({p1} ∧ {p2} → {head})"
        rules_out.append(rule)
        used_heads.add(head)

    return pred_list, rules_out

# ——— PCFG Generator 类，增强类型控制 ———

class PCFGGenerator:
    def __init__(self, predicates: List[str], rules: List[str], x0: str = "x0", y0: str = "y0"):
        self.predicates = predicates
        self.rules = rules
        self.x0 = x0
        self.y0 = y0
        self.prod_counts = Counter()
        self.nonterminal_counts = Counter()
        self._train()

    def _train(self):
        for rule in self.rules:
            m = re.match(r'Q\[.*?\]\((.*)→(.*)\)', rule)
            if not m:
                continue
            premise_str = m.group(1).strip()
            conclusion = m.group(2).strip()
            premise_preds = [p.strip() for p in premise_str.split("∧") if p.strip()]

            # 统计 PREMIS E产生式
            self.prod_counts[('PREMISE', tuple(premise_preds))] += 1
            self.nonterminal_counts['PREMISE'] += 1

            # 统计 S → (PREMISE, conclusion)
            self.prod_counts[('S', ('PREMISE', conclusion))] += 1
            self.nonterminal_counts['S'] += 1

    def _sample_premise(self, min_len=2, max_len=4) -> List[str]:
        # 根据类型选择前提谓词
        type_buckets = defaultdict(list)
        for p in self.predicates:
            type_buckets[classify_predicate(p)].append(p)

        # 随机决定前提长度
        plen = random.randint(min_len, min(max_len, len(self.predicates)))
        premise = []
        used = set()
        # 尝试在不同类型中采样
        types_list = list(type_buckets.keys())
        for _ in range(plen):
            # 随机选类型，再选一个谓词
            t = random.choice(types_list)
            bucket = type_buckets.get(t)
            if bucket:
                cand = [q for q in bucket if q not in used]
                if cand:
                    choice = random.choice(cand)
                    premise.append(choice)
                    used.add(choice)
                    continue
            # 如果类型桶空或没剩余，随机选一个剩下的
            rem = [q for q in self.predicates if q not in used]
            if rem:
                choice = random.choice(rem)
                premise.append(choice)
                used.add(choice)

        return premise

    def _sample_rule(self, min_len=2, max_len=4) -> str:
        premise = self._sample_premise(min_len, max_len)

        # 倾向某些类型作为结论
        candidates = [
            p for p in self.predicates
            if classify_predicate(p) in ("ml_pred", "contra", "linkage", "edge_attr", "node_attr")
        ]
        if not candidates:
            conclusion = random.choice(self.predicates)
        else:
            conclusion = random.choice(candidates)

        # 确保结论不在前提中
        if conclusion in premise:
            rem = [q for q in self.predicates if q not in premise]
            if rem:
                conclusion = random.choice(rem)

        # ✅ 使用类属性 self.x0 和 self.y0
        rule = f"Q[{self.x0},{self.y0}] ({' ∧ '.join(premise)} → {conclusion})"
        return rule

    def generate_rules(self, num: int, min_len=2, max_len=4) -> List[str]:
        rules_generated = set()
        attempts = 0
        max_attempts = num * 5
        while len(rules_generated) < num and attempts < max_attempts:
            r = self._sample_rule(min_len, max_len)
            rules_generated.add(r)
            attempts += 1
        return list(rules_generated)

# ——— 主流程示例 ———

if __name__ == "__main__":
    # 加载训练数据
    train_preds, train_rules = load_predicates_and_rules("./star_outputs/dual_star_graph_1_Emily_Clark_rico.txt")
    with open("../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # 假设你有 data, doc1, doc2, x0, y0 已准备好
    # 举例：
    # from somewhere import load_graph_triples
    # data = load_graph_triples("../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json")
    # doc1, doc2 = "docA", "docB"
    # x0, y0 = "x_node", "y_node"

    # 为测试临时使用 train_preds 作为 predicates_new
    predicates_new, rules_new = generate_predicates_and_rules(
        data=train_data,   # 替换为你的 data
        doc1="doc1",
        doc2="doc2",
        x0="x0",
        y0="y0",
        ml_threshold=0.5,
        max_rules=10
    )

    pcfg = PCFGGenerator(predicates_new, train_rules)
    sampled_rules = pcfg.generate_rules(num=20, min_len=2, max_len=4)

    print("🧠 新生成的谓词：")
    for p in predicates_new:
        print(p)

    with open("./star_outputs/generated_predicates_pcfg.txt", "w", encoding="utf-8") as f:
        for p in predicates_new:
            f.write(p + "\n")

    print("\n🧠 新生成的规则：")
    for r in sampled_rules:
        print(r)

    with open("./star_outputs/generated_rules_pcfg.txt", "w", encoding="utf-8") as f:
        for r in sampled_rules:
            f.write(r + "\n")



