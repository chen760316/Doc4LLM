"""
从训练规则中学习逻辑结构／规则的概率分布（模板学习），再基于此生成新的规则

✅ 对比：PCFG vs 贝叶斯网络（BayesNet）
特点	PCFG	贝叶斯网络（BayesNet）
结构	使用上下文无关的产生式（A → B C）	使用有向图表示变量依赖关系
概率来源	规则频率	条件概率表 CPT
表达能力	生成语法结构	建模变量之间的条件依赖
适合什么	结构生成（如语法树）	表达复杂因果关系或依赖关系

✅ 总结：关键功能标注
功能	是否实现	代码位置	注释
结构学习	✅ 是	BayesianRuleGenerator._learn_structure()	从已有规则中提取 premise → conclusion 依赖关系，构建网络结构
参数学习（概率学习）	✅ 是	BayesianRuleGenerator._learn_parameters()	基于频率统计生成 CPT 表
基于结构+CPT生成新规则	✅ 是	BayesianRuleGenerator.sample_rule()	根据条件概率按权重采样生成新规则
批量生成新规则	✅ 是	BayesianRuleGenerator.generate_rules()	多次采样生成多个规则
"""

import random
import re
import json
from collections import Counter, defaultdict
from typing import List, Tuple
from itertools import combinations

class BayesianRuleGenerator:
    def __init__(self, predicates: List[str], rules: List[str]):
        self.predicates = predicates
        self.rules = rules
        self.structure = defaultdict(list)  # 贝叶斯结构：结论 <- premise
        self.cpt = defaultdict(Counter)     # 条件概率表 CPT
        self._learn_structure()
        self._learn_parameters()

    """
    从已有规则中提取出规则的结构（谁依赖谁）
    ✅ 说明：
    规则形如 Q[x0,y0] (A ∧ B → C)，被解析为：
    premise：A, B
    conclusion：C
    存入 self.structure[conclusion] = list of premise tuples
    🔁 这就是一个**“贝叶斯网络结构”的学习过程**，建成的是一个简化版有向图（conclusion <- premise）
    """

    def _learn_structure(self):
        # 对于每个规则，从 premise 到 conclusion 添加依赖
        for rule in self.rules:
            m = re.match(r'Q\[.*?\]\((.*)→(.*)\)', rule)
            if not m:
                continue
            premise = tuple(sorted(p.strip() for p in m.group(1).split("∧") if p.strip()))
            conclusion = m.group(2).strip()
            self.structure[conclusion].append(premise)

    """
    统计每个 conclusion 给定不同 premise 时出现的频率，计算条件概率 P(conclusion | premise)
    ✅ 说明：
    使用 Counter 统计各前提组合的频率
    再进行归一化，得到条件概率
    存入 self.cpt：条件概率表（CPT, Conditional Probability Table）
    """

    def _learn_parameters(self):
        # 对结构中的每个 conclusion，统计其前提组合的频率
        for conclusion, premises in self.structure.items():
            for premise in premises:
                self.cpt[conclusion][premise] += 1

        # Normalize to get P(conclusion | premise)
        for conclusion in self.cpt:
            total = sum(self.cpt[conclusion].values())
            for premise in self.cpt[conclusion]:
                self.cpt[conclusion][premise] /= total

    """
    功能： 根据训练得到的结构 + CPT，采样生成规则
    ✅ 说明：
    从谓词集合中随机采样一些谓词作为“前提”
    然后在 self.cpt 里查找匹配的结论（其条件前提是当前前提的子集）
    用条件概率进行加权采样
    生成新的规则
    """

    def sample_rule(self, min_len=2, max_len=4) -> str:
        # 随机采样 premise
        selected = random.sample(self.predicates, random.randint(min_len, max_len))
        selected_key = tuple(sorted(selected))

        # 找所有结论满足 premise 的
        matching = []
        for conclusion, cond_probs in self.cpt.items():
            for cond_premise in cond_probs:
                if set(cond_premise).issubset(set(selected_key)):
                    matching.append((conclusion, cond_probs[cond_premise]))

        if not matching:
            # fallback
            conclusion = random.choice(self.predicates)
        else:
            # 按概率采样结论
            conclusions, probs = zip(*matching)
            conclusion = random.choices(conclusions, weights=probs, k=1)[0]

        rule = f"Q[x0,y0] ({' ∧ '.join(selected)} → {conclusion})"
        return rule

    """
    批量生成规则
    调用上面的 sample_rule()，生成 num 条新规则。
    """

    def generate_rules(self, num=10, min_len=2, max_len=4):
        rules = []
        for _ in range(num):
            r = self.sample_rule(min_len, max_len)
            rules.append(r)
        return rules

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

if __name__ == "__main__":
    # —— 步骤 1: 加载已有谓词和规则（训练用） ——
    train_preds, train_rules = load_predicates_and_rules("./star_outputs/dual_star_graph_1_Emily_Clark_rico.txt")

    # —— 步骤 2: 加载图数据（用于生成新谓词） ——
    with open("../star_json/dual_graph/dual_star_graph_1_Emily_Clark.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # —— 步骤 3: 使用图数据生成新的谓词和规则候选 ——
    predicates_new, _ = generate_predicates_and_rules(
        data=train_data,
        doc1="doc1",
        doc2="doc2",
        x0="x0",
        y0="y0",
        ml_threshold=0.5,
        max_rules=10  # 这里只是初始规则，不是最终生成的
    )

    print("📊 新生成的谓词数量:", len(predicates_new))

    # —— 步骤 4: 用已有规则（train_rules）+新谓词训练贝叶斯规则生成器 ——
    bayes_gen = BayesianRuleGenerator(predicates=predicates_new, rules=train_rules)

    # —— 步骤 5: 采样生成规则 ——
    new_rules = bayes_gen.generate_rules(num=20, min_len=2, max_len=4)

    # —— 步骤 6: 打印结果 ——
    print("\n🧠 新生成的规则（来自 BayesNet 模型）：")
    for r in new_rules:
        print(r)

    # （可选）保存到文件
    with open("./star_outputs/generated_rules_bayes.txt", "w", encoding="utf-8") as f:
        for r in new_rules:
            f.write(r + "\n")
