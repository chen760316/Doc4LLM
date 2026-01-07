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
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
import os

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
        selected = random.sample(list(self.predicates), random.randint(min_len, max_len))
        selected_key = tuple(sorted(selected))  # 排序后转换为元组，用作条件

        # 找所有结论满足 premise 的条件
        matching = []
        for conclusion, cond_probs in self.cpt.items():
            for cond_premise, prob in cond_probs.items():
                # 判断 premise 是否是条件的子集
                if set(cond_premise).issubset(set(selected_key)):
                    matching.append((conclusion, prob))

        if not matching:
            # 如果没有匹配的结论，回退：随机选择一个谓词作为结论
            conclusion = random.choice(list(self.predicates))
        else:
            # 按概率采样结论
            conclusions, probs = zip(*matching)  # 解压 matching 元组
            conclusion = random.choices(conclusions, weights=probs, k=1)[0]

        # 构造规则字符串
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

def get_contradiction_type(s1, s2):
    # 将system_prompt和实际文本作为用户输入传递给LLM
    rules_prompt = f"""
    You are an expert in detecting contradiction types between two sentences.

    Given two sentences, identify if they are contradictory and, if so, classify the contradiction into one of the following 7 types. Each contradiction type is explained with multiple detailed examples to clarify the scope of each category:

    1. **Factual**: Requires knowledge of the external world to verify contradictions (external knowledge bases are needed for detection).
       Example: 
       - Sentence 1: "The Nobel Prize was indeed established in 1901 to commemorate Alfred Nobel."
       - Sentence 2: "Hinton received a world-renowned scientific award in 2024, which was established in 1901 to honor Shannon's contributions to the field of information."
       Contradiction type: **Factual**, because the Nobel Prize was established in 1901 to honor Alfred Nobel, not Shannon, and requires external knowledge to confirm the misattribution.

    2. **Content**: Contradictions caused by altering one or more attributes of an event or entity. This includes:
       - **Opinion/Emotion**: A contradiction involving differing emotional reactions or opinions.
         Example:
         - Sentence 1: "William was engaged while reading the paper on computer architecture."
         - Sentence 2: "William found the content of the paper on computer architecture to be extremely boring."
         Contradiction type: **Content (Opinion/Emotion)**, because the contradiction arises from different emotional responses to the same paper (engagement vs. boredom).

    3. **Causal**: Contradictions arising from a mismatch between cause and effect (must involve a clear causal relationship).
       Example:
       - Sentence 1: "David’s video, which was published on YouTube in 2024, was permanently removed that year due to violence."
       - Sentence 2: "Taylor came across this video published by David while browsing YouTube in 2025."
       Contradiction type: **Causal**, because the video was supposed to be removed in 2024, but it is still available in 2025.

    4. **Negation**: Contradictions resulting from negating the original sentence (must involve explicit negation, such as antonyms or negative words like "do not," "is not").
       Example:
       - Sentence 1: "On April 20, 2025, the router experienced a brief routing disruption at 2:00 PM."
       - Sentence 2: "The router did not report any routing disruptions throughout April 20, 2025."
       Contradiction type: **Negation**, because the second sentence explicitly negates the occurrence of the routing disruption mentioned in the first sentence.

    5. **Relation**: Contradictions arising from descriptions of two mutually exclusive relationships between entities (must involve conflicts in relationships between entities), including reversed, missing, or mismatched relationships.
       Example:
       - Sentence 1: "The owner of the brand store, Howard, and his wife Sheila have a harmonious relationship after marriage."
       - Sentence 2: "Howard and his wife Emma have a beloved little daughter named Sheila."
       Contradiction type: **Relation**, because the first sentence implies Howard is married to Sheila, while the second sentence implies he is married to Emma, creating a contradiction in the relationships.

    6. **Numeric**: Contradictions due to mismatched numbers or numbers falling outside an acceptable range.
       Example:
       - Sentence 1: "Howard’s brand store sold 29 pairs of women’s high heels on June 15, 2025."
       - Sentence 2: "Howard is concerned about the lack of any sales on June 15, 2025."
       Contradiction type: **Numeric**, because one sentence claims sales of 29 pairs of high heels, while the other suggests there were no sales at all.

    7. **Temporal**: Contradictions caused by changes or updates in knowledge over time, affecting temporal order.
       Example:
       - Sentence 1: "Around 1860, Nobel and other researchers believed that nitroglycerin could not treat heart disease."
       - Sentence 2: "After Nobel’s death, further research revealed that nitroglycerin is effective for treating heart disease."
       Contradiction type: **Temporal**, because the belief about nitroglycerin changed over time, from it being ineffective in the 1860s to being effective after Nobel's death.

    If the sentences are contradictory, output ONLY the contradiction type: one of ["Factual", "Content", "Causal", "Negation", "Relation", "Numeric", "Temporal"]. Include the reasoning for the contradiction after the type. If the sentences are **NOT contradictory**, output the phrase "No Contradiction" along with the reason why they do not contradict.

    Sentence 1: {s1}
    Sentence 2: {s2}
    """

    system_message = "You are a helpful assistant that analyzes contradictions between sentences."

    # 调用 LLM 进行判断
    response = llm.generate(user_message=rules_prompt, system_message=system_message)

    # 解析返回的响应，提取矛盾类型部分
    contradiction_type = re.split(r'[,:.]', response)[0].strip()  # 按逗号、冒号、句号分割并取第一个部分

    # 返回矛盾类型
    return contradiction_type

def generate_predicates_with_llm(graph_json_path: str, llm, num_predicates: int = 100) -> List[str]:
    predicates = []
    entity_predicates = []
    edge_predicates = []
    meta_predicates = []

    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    # 计算每种类型的最大数量
    max_entity = max(1, num_predicates // 4)  # 保证至少1个实体谓词
    max_edge = max(1, num_predicates // 4)  # 保证至少1个边谓词
    max_meta = num_predicates - (max_entity + max_edge)  # 剩余给meta类型

    # Entity-level predicates
    for data in graph_data:
        # 确保 node_attrs 存在并且非空
        node_attrs = data.get("node_attrs", {})
        multi_hop_node_attrs = data.get("multi_hop_node_attrs", {})

        # 处理 node_attrs 中的节点属性
        for node, attrs in node_attrs.items():
            # 属性比较 predicates
            for attr, value in attrs.items():
                if attr == "text_span":
                    continue  # 跳过 text_span 属性
                if len(entity_predicates) < max_entity:
                    entity_predicates.append(f'{node}.{attr} = "{value}"')

                    # 不等式比较
                    if isinstance(value, (int, float)):
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} != {value}')
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} < {value}')
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} > {value}')
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} <= {value}')
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} >= {value}')
                    elif isinstance(value, str):
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} != "{value}"')

        # 处理 multi_hop_node_attrs 中的节点属性
        for node, attrs in multi_hop_node_attrs.items():
            # 属性比较 predicates
            for attr, value in attrs.items():
                if attr == "text_span":
                    continue  # 跳过 text_span 属性
                if len(entity_predicates) < max_entity:
                    entity_predicates.append(f'{node}.{attr} = "{value}"')

                    # 不等式比较
                    if isinstance(value, (int, float)):
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} != {value}')
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} < {value}')
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} > {value}')
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} <= {value}')
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} >= {value}')
                    elif isinstance(value, str):
                        if len(entity_predicates) < max_entity:
                            entity_predicates.append(f'{node}.{attr} != "{value}"')

    # Edge-level predicates
    for data in graph_data:
        # 确保 edges_attrs 存在并且非空
        edge_attrs = data.get("edges_attrs", {})
        multi_hop_edges_attrs = data.get("multi_hop_edges_attrs", {})

        # 处理普通的边属性
        for edge, attrs in edge_attrs.items():
            # 边属性比较 predicates
            for attr, value in attrs.items():
                if attr == "text_span":
                    continue  # 跳过 text_span 属性
                if isinstance(value, (int, float)) and len(edge_predicates) < max_edge:
                    edge_predicates.append(f'{edge}.{attr} = {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} != {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} < {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} > {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} <= {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} >= {value}')
                elif isinstance(value, str) and len(edge_predicates) < max_edge:
                    edge_predicates.append(f'{edge}.{attr} = "{value}"')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} != "{value}"')

        # 处理 multi_hop_edges_attrs 中的边属性
        for edge, attrs in multi_hop_edges_attrs.items():
            # 边属性比较 predicates
            for attr, value in attrs.items():
                if attr == "text_span":
                    continue  # 跳过 text_span 属性
                if isinstance(value, (int, float)) and len(edge_predicates) < max_edge:
                    edge_predicates.append(f'{edge}.{attr} = {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} != {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} < {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} > {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} <= {value}')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} >= {value}')
                elif isinstance(value, str) and len(edge_predicates) < max_edge:
                    edge_predicates.append(f'{edge}.{attr} = "{value}"')
                    if len(edge_predicates) < max_edge:
                        edge_predicates.append(f'{edge}.{attr} != "{value}"')

    # Meta-level predicates
    for data in graph_data:
        # 确保 node_attrs 和 edges_attrs 不为空
        node_attrs = data.get("node_attrs", {})
        edge_attrs = data.get("edges_attrs", {})
        multi_hop_node_attrs = data.get("multi_hop_node_attrs", {})
        multi_hop_edges_attrs = data.get("multi_hop_edges_attrs", {})

        # 如果 node_attrs 存在并且非空，执行实体之间的谓词
        if node_attrs:
            # M_NLI, M_NLU, M_TEM between entities
            for node1, attrs1 in node_attrs.items():
                # 拼接节点的所有属性
                node1_attrs_str = ', '.join(f"{key}={value}" for key, value in attrs1.items() if key != "text_span")
                for node2, attrs2 in node_attrs.items():
                    if node1 != node2 and len(meta_predicates) < max_meta:
                        # 拼接节点2的所有属性
                        node2_attrs_str = ', '.join(f"{key}={value}" for key, value in attrs2.items() if key != "text_span")

                        # 添加 M_NLI, M_NLU, M_TEM 谓词
                        meta_predicates.append(f'M_NLI({node1}({node1_attrs_str}), {node2}({node2_attrs_str}))')
                        if len(meta_predicates) < max_meta:
                            meta_predicates.append(f'M_NLU({node1}({node1_attrs_str}), {node2}({node2_attrs_str}))')
                        if len(meta_predicates) < max_meta:
                            meta_predicates.append(f'M_TEM({node1}({node1_attrs_str}), {node2}({node2_attrs_str}))')

        # 如果 multi_hop_node_attrs 存在并且非空，执行多跳实体之间的谓词
        if multi_hop_node_attrs:
            # M_NLI, M_NLU, M_TEM between multi-hop entities
            for node1, attrs1 in multi_hop_node_attrs.items():
                # 拼接节点的所有属性
                node1_attrs_str = ', '.join(f"{key}={value}" for key, value in attrs1.items() if key != "text_span")
                for node2, attrs2 in multi_hop_node_attrs.items():
                    if node1 != node2 and len(meta_predicates) < max_meta:
                        # 拼接节点2的所有属性
                        node2_attrs_str = ', '.join(f"{key}={value}" for key, value in attrs2.items() if key != "text_span")

                        # 添加 M_NLI, M_NLU, M_TEM 谓词
                        meta_predicates.append(f'M_NLI({node1}({node1_attrs_str}), {node2}({node2_attrs_str}))')
                        if len(meta_predicates) < max_meta:
                            meta_predicates.append(f'M_NLU({node1}({node1_attrs_str}), {node2}({node2_attrs_str}))')
                        if len(meta_predicates) < max_meta:
                            meta_predicates.append(f'M_TEM({node1}({node1_attrs_str}), {node2}({node2_attrs_str}))')

        # 如果 edge_attrs 存在并且非空，执行边之间的谓词
        if edge_attrs:
            # M_NLI, M_NLU, M_TEM between edges
            for edge1, attrs1 in edge_attrs.items():
                # 拼接边的所有属性
                edge1_attrs_str = ', '.join(f"{key}={value}" for key, value in attrs1.items())
                for edge2, attrs2 in edge_attrs.items():
                    if edge1 != edge2 and len(meta_predicates) < max_meta:
                        # 拼接边2的所有属性
                        edge2_attrs_str = ', '.join(f"{key}={value}" for key, value in attrs2.items())

                        # 添加 M_NLI, M_NLU, M_TEM 谓词
                        meta_predicates.append(f'M_NLI({edge1}({edge1_attrs_str}), {edge2}({edge2_attrs_str}))')
                        if len(meta_predicates) < max_meta:
                            meta_predicates.append(f'M_NLU({edge1}({edge1_attrs_str}), {edge2}({edge2_attrs_str}))')
                        if len(meta_predicates) < max_meta:
                            meta_predicates.append(f'M_TEM({edge1}({edge1_attrs_str}), {edge2}({edge2_attrs_str}))')

        # 如果 multi_hop_edges_attrs 存在并且非空，执行多跳边之间的谓词
        if multi_hop_edges_attrs:
            # M_NLI, M_NLU, M_TEM between multi-hop edges
            for edge1, attrs1 in multi_hop_edges_attrs.items():
                # 拼接边的所有属性
                edge1_attrs_str = ', '.join(f"{key}={value}" for key, value in attrs1.items())
                for edge2, attrs2 in multi_hop_edges_attrs.items():
                    if edge1 != edge2 and len(meta_predicates) < max_meta:
                        # 拼接边2的所有属性
                        edge2_attrs_str = ', '.join(f"{key}={value}" for key, value in attrs2.items())

                        # 添加 M_NLI, M_NLU, M_TEM 谓词
                        meta_predicates.append(f'M_NLI({edge1}({edge1_attrs_str}), {edge2}({edge2_attrs_str}))')
                        if len(meta_predicates) < max_meta:
                            meta_predicates.append(f'M_NLU({edge1}({edge1_attrs_str}), {edge2}({edge2_attrs_str}))')
                        if len(meta_predicates) < max_meta:
                            meta_predicates.append(f'M_TEM({edge1}({edge1_attrs_str}), {edge2}({edge2_attrs_str}))')

    # 遍历graph_data，生成S(x, s)和S(ex, s)谓词
    # 用于存储节点/边的文本片段，方便后续使用
    node_texts = {}
    edge_texts = {}

    # 遍历graph_data，生成S(x, s)和S(ex, s)谓词
    for idx, data in enumerate(graph_data):
        # 获取节点和边属性
        node_attrs = data.get("node_attrs", {})
        edge_attrs = data.get("edges_attrs", {})
        multi_hop_node_attrs = data.get("multi_hop_node_attrs", {})
        multi_hop_edges_attrs = data.get("multi_hop_edges_attrs", {})

        # 选择 doc1 或 doc2，控制文本来源
        if idx % 2 == 0:
            doc = doc1  # 偶数索引使用 doc1
        else:
            doc = doc2  # 奇数索引使用 doc2

        # 如果 node_attrs 不为空，则处理节点相关的 S(x, s) 谓词
        if node_attrs:
            for node, attrs in node_attrs.items():
                text_span = attrs.get("text_span")
                if text_span and len(meta_predicates) < max_meta:
                    # 提取文本片段
                    start, end = text_span.get("start"), text_span.get("end")
                    s = doc[start:end]

                    # 获取节点的其他属性并与文本拼接
                    additional_info = ' '.join(f"{key}={value}" for key, value in attrs.items() if key != 'text_span')
                    s = f"{s} ({additional_info})" if additional_info else f"{node}: {s}"

                    # 存储节点的文本片段
                    node_texts[node] = s
                    meta_predicates.append(f'S({node}, "{s}")')

        # 如果 multi_hop_node_attrs 不为空，则处理多跳节点相关的 S(x, s) 谓词
        if multi_hop_node_attrs:
            for node, attrs in multi_hop_node_attrs.items():
                text_span = attrs.get("text_span")
                if text_span and len(meta_predicates) < max_meta:
                    # 提取文本片段
                    start, end = text_span.get("start"), text_span.get("end")
                    s = doc[start:end]

                    # 获取节点的其他属性并与文本拼接
                    additional_info = ' '.join(f"{key}={value}" for key, value in attrs.items() if key != 'text_span')
                    s = f"{s} ({additional_info})" if additional_info else f"{node}: {s}"

                    # 存储节点的文本片段
                    node_texts[node] = s
                    meta_predicates.append(f'S({node}, "{s}")')

        # 如果 edge_attrs 不为空，则处理边相关的 S(ex, s) 谓词
        if edge_attrs:
            for edge, attrs in edge_attrs.items():
                text_span = attrs.get("text_span")
                if text_span and len(meta_predicates) < max_meta:
                    # 提取文本片段
                    start, end = text_span.get("start"), text_span.get("end")
                    s = doc[start:end]

                    # 存储边的文本片段
                    edge_texts[edge] = s
                    meta_predicates.append(f'S({edge}, "{s}")')

        # 如果 multi_hop_edges_attrs 不为空，则处理多跳边相关的 S(ex, s) 谓词
        if multi_hop_edges_attrs:
            for edge, attrs in multi_hop_edges_attrs.items():
                text_span = attrs.get("text_span")
                if text_span and len(meta_predicates) < max_meta:
                    # 提取文本片段
                    start, end = text_span.get("start"), text_span.get("end")
                    s = doc[start:end]

                    # 存储边的文本片段
                    edge_texts[edge] = s
                    meta_predicates.append(f'S({edge}, "{s}")')

    # T(s, s', tau)，\tau(x.A, y.B)，\tau(ex.A, ey.B) for contradiction detection
    for data in graph_data:
        # 获取节点和边的属性
        node_attrs = data.get("node_attrs", {})
        edge_attrs = data.get("edges_attrs", {})

        # 确保 node_attrs 和 edge_attrs 不为空
        if node_attrs or edge_attrs:
            # 处理节点之间的矛盾检测
            for node1, attrs1 in node_attrs.items():
                for node2, attrs2 in node_attrs.items():
                    if node1 != node2 and len(meta_predicates) < max_meta:
                        # 获取节点文本片段 s 和 s'
                        s1 = node_texts.get(node1, "")
                        s2 = node_texts.get(node2, "")

                        # 获取节点位置标识符
                        node1_pos = attrs1.get("position", "")
                        node2_pos = attrs2.get("position", "")

                        # 确保两个节点有对应的文本片段
                        if s1 and s2:
                            # 使用 LLM 判断矛盾类型
                            contradiction_type = get_contradiction_type(s1, s2)

                            if "Contradiction" in contradiction_type:
                                # 添加矛盾类型的谓词
                                meta_predicates.append(f'T({s1}, {s2}, {contradiction_type})')
                                # 添加矛盾发生的具体位置谓词
                                meta_predicates.append(f'{contradiction_type}({node1}.{node1_pos}, {node2}.{node2_pos})')

            # 处理边之间的矛盾检测
            for edge1, attrs1 in edge_attrs.items():
                for edge2, attrs2 in edge_attrs.items():
                    if edge1 != edge2 and len(meta_predicates) < max_meta:
                        # 获取边文本片段 s 和 s'
                        s1 = edge_texts.get(edge1, "")
                        s2 = edge_texts.get(edge2, "")

                        # 获取边位置标识符
                        edge1_pos = attrs1.get("position", "")
                        edge2_pos = attrs2.get("position", "")

                        # 确保两个边有对应的文本片段
                        if s1 and s2:
                            # 使用 LLM 判断矛盾类型
                            contradiction_type = get_contradiction_type(s1, s2)

                            if "Contradiction" in contradiction_type:
                                # 添加矛盾类型的谓词
                                meta_predicates.append(f'T({s1}, {s2}, {contradiction_type})')
                                # 添加矛盾发生的具体位置谓词
                                meta_predicates.append(f'{contradiction_type}({edge1}.{edge1_pos}, {edge2}.{edge2_pos})')

            # 处理节点与边之间的矛盾检测
            for node, attrs1 in node_attrs.items():
                for edge, attrs2 in edge_attrs.items():
                    if len(meta_predicates) < max_meta:
                        # 获取节点和边的文本片段 s 和 s'
                        s1 = node_texts.get(node, "")
                        s2 = edge_texts.get(edge, "")

                        # 获取节点和边位置标识符
                        node_pos = attrs1.get("position", "")
                        edge_pos = attrs2.get("position", "")

                        # 确保节点和边都有对应的文本片段
                        if s1 and s2:
                            # 使用 LLM 判断矛盾类型
                            contradiction_type = get_contradiction_type(s1, s2)

                            if "Contradiction" in contradiction_type:
                                # 添加矛盾类型的谓词
                                meta_predicates.append(f'T({s1}, {s2}, {contradiction_type})')
                                # 添加矛盾发生的具体位置谓词
                                meta_predicates.append(f'{contradiction_type}({node}.{node_pos}, {edge}.{edge_pos})')

    # 合并所有谓词，保持顺序
    predicates = list(entity_predicates) + list(edge_predicates) + list(meta_predicates)
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

    # 设定你的目录
    OUTPUT_DIR = "dual_star/bayesnet"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # —— 步骤 1: 遍历所有 JSON 文件 ——
    DUAL_STAR_JSON_DIR = "./dual_star_json_with_attrs"  # 更新为新目录
    json_files = [os.path.join(DUAL_STAR_JSON_DIR, f) for f in os.listdir(DUAL_STAR_JSON_DIR) if f.endswith(".json")]

    if not json_files:
        print(f"⚠️ 没有找到任何 JSON 文件在目录: {DUAL_STAR_JSON_DIR}")
        exit(1)

    # —— 步骤 2: 加载图数据（用于生成新谓词） ——
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)

        # —— 步骤 3: 加载图数据并提取信息 ——
        data = load_dual_star_json(json_file)
        x0, y0 = find_dual_star_centers(data)

        # 推测对应文本路径，比如 text1/2 位于 texts 文件夹中
        # 确保这些文件存在或按规则命名
        text1_path = "dense_text1.txt"
        text2_path = "dense_text2.txt"

        if not os.path.exists(text1_path) or not os.path.exists(text2_path):
            print(f"⚠️ 缺失对应文本文件: {text1_path} 或 {text2_path}")
            continue

        doc1 = load_text_file(text1_path)
        doc2 = load_text_file(text2_path)

        preds, rules = generate_predicates_with_llm(json_file, llm)
        print(f"📊 文件 '{json_file}' 新生成的谓词数量:", len(preds))

        # —— 步骤 4: 用已有规则（train_rules）+新谓词训练贝叶斯规则生成器 ——
        bayes_gen = BayesianRuleGenerator(predicates=preds, rules=rules)

        # —— 步骤 5: 采样生成规则 ——
        new_rules = bayes_gen.generate_rules(num=20, min_len=2, max_len=4)

        # —— 步骤 6: 打印结果 ——
        print(f"\n🧠 文件 '{json_file}' 新生成的规则（来自 BayesNet 模型）：")
        for r in new_rules:
            print(r)

        # 设置输出文件路径
        output_file_path = os.path.join(OUTPUT_DIR, f"generated_rules_bayes_{os.path.basename(json_file)}.txt")

        # 假设 new_rules 是一个包含规则的列表
        with open(output_file_path, "w", encoding="utf-8") as f:
            for r in new_rules:
                f.write(r + "\n")

        print(f"文件 '{json_file}' 结果已保存到: {output_file_path}")