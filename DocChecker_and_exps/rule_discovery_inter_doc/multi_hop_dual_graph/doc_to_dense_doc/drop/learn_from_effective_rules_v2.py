"""
执行效率太低
z没有利用有效规则进行规则学习，暂时useless
"""
import os
import json
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from collections import defaultdict
from typing import List
import re

# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

def load_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# 设定你的目录
text1_path = "../dense_text1.txt"
text2_path = "../dense_text2.txt"
if not os.path.exists(text1_path) or not os.path.exists(text2_path):
    print(f"⚠️ 缺失对应文本文件: {text1_path} 或 {text2_path}")

doc1 = load_text_file(text1_path)
doc2 = load_text_file(text2_path)

# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# 设定你的目录
llm_folder = './dual_star/llm_evaluations_v3/effective_rules'  # 用来加载已有有效规则的目录
json_data_folder = './dual_star_json_with_attrs'  # 用来加载JSON数据的目录
contradictions_file = '../dense_text_cross.txt'  # 用来加载矛盾类型的文件
output_folder = './dual_star/learn_from_effective_rules'  # 输出评估结果的文件夹路径

os.makedirs(output_folder, exist_ok=True)

def load_contradictions(file_path: str) -> List[str]:
    """加载矛盾类型文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

# 加载矛盾类型
contradictions = load_contradictions(contradictions_file)

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

    # 确保总数量不超过num_predicates
    return predicates

def build_rule_generation_prompt(rico_predicates: List[str], example_rules: List[str]) -> str:
    """构建生成规则的提示"""
    return f"""You are an expert in logical reasoning and contradiction detection.

Your task is to generate RICO-style rules based on the given list of predicates.

Each rule should follow the format:
Q[x0, y0](P1 ∧ P2 ∧ ... → P_out)

### Rule Constraints:

1. Each rule must include between **2 and 6 predicate conditions** in the premise.
2. You must generate a **diverse set of rules** with varying premise lengths — specifically, make sure to include rules whose premises have 2, 3, 4, 5, and 6 predicate conditions.
3. The number of rules for each premise length should be **as evenly distributed as possible** — for example, if you are generating 20 rules, try to generate about 4 rules for each premise length.
4. Each rule must have **exactly one predicate in the conclusion**.
5. The **conclusion predicate must not be identical to any of the predicates in the premise**. That is, avoid reusing the same predicate in both the premise and the conclusion of a rule.

Use only the provided predicates. Use them in combinations to create **reasonable and minimal** logical implications. Try to make each rule capture a plausible constraint, contradiction, or implication.

--- Examples ---

Rules:
{chr(10).join(example_rules)}

--- Predicates ---
{chr(10).join(rico_predicates)}

--- Now write exactly 30 new rules below ---
Rules:"""


def extract_predicates(file_path: str) -> list:
    """从文件中提取谓词部分，并去除空行"""
    predicates = []

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 未找到")

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        is_predicates_section = False
        for line in file:
            # 开始提取谓词部分
            if line.strip() == "Predicates:":
                is_predicates_section = True
                continue  # 跳过 "Predicates:" 本身

            # 停止提取谓词部分
            if line.strip() == "Rules:":
                break

            # 如果在谓词部分，则将每行添加到列表，且去除空行
            if is_predicates_section:
                stripped_line = line.strip()  # 去除行首尾空白字符
                if stripped_line:  # 如果这一行不为空，则加入谓词列表
                    predicates.append(stripped_line)

    return predicates


def generate_rico_predicates_and_rules_with_learning(
        graph_json_path: str,
        output_path: str,
        llm,
        contradictions: List[str],
        num_predicates: int = 100,
        num_rules: int = 100,
        previous_rules: List[str] = []
):
    """生成谓词和规则，并根据矛盾类型评估规则的有效性，同时允许从历史规则中学习"""

    # === Step 1: Generate predicates from graph ===
    # predicates = generate_predicates_with_llm(graph_json_path, llm, num_predicates=num_predicates)
    # 候选实现如下：
    predicates = extract_predicates(graph_json_path)

    # === Step 2: Generate rules based on predicates and previous rules ===
    # Combine new predicates with previous rules to guide new rule generation
    rules_prompt = build_rule_generation_prompt(predicates + previous_rules, example_rules=previous_rules)

    # Enhance learning from past rules by incorporating patterns from successful/unsuccessful rules
    system_message = (
        "You are a natural language inference engine that generates RICO-style logical rules "
        "from given predicates. Each rule should follow the format Q[x0, y0](P1 ∧ P2 ∧ ... → P_out). "
        "Refer to historical rules to learn patterns of valid and invalid rules. "
        "Generate rules based on both the new predicates and your understanding of prior rules. "
        "If similar patterns exist in historical rules, apply them."
    )

    response = llm.generate(
        user_message=rules_prompt,
        system_message=system_message
    )
    rules = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # === Step 3: Evaluate rules with contradictions and history learning ===
    valid_rules = []
    for rule in rules:
        # Evaluate the rule against contradictions first
        is_valid = evaluate_rule_against_contradictions(rule, contradictions)
        if is_valid:
            valid_rules.append(rule)

    # === Step 4: Save everything ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Predicates:\n")
        for p in predicates:
            f.write(p + "\n")
        f.write("\nRules:\n")
        for r in valid_rules:
            f.write(r + "\n")

    print(f"\n✅ Successfully saved {len(predicates)} predicates and {len(valid_rules)} valid rules to: {output_path}")

    # Return the new valid rules to update the previous rules for further learning
    return valid_rules


def load_previous_rules_from_folder(folder_path: str) -> List[str]:
    """从指定文件夹加载历史规则"""
    previous_rules = []
    rule_start_pattern = re.compile(r'Rules:\s*')

    # 遍历文件夹中的每个文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # 查找 Rules: 后的部分，提取规则内容
                match = rule_start_pattern.search(content)
                if match:
                    # 从 Rules: 后面开始提取规则部分
                    rules_section = content[match.end():].strip()

                    # 处理规则部分，去掉头部描述信息、空行等非规则内容
                    # 如果文件中有以 "Rule" 开头的行，假设这是实际的规则部分
                    rule_lines = []
                    for line in rules_section.splitlines():
                        # 假设规则是以 Q[开头
                        if line.startswith("Q["):
                            rule_lines.append(line.strip())

                    # 将提取到的规则加入到历史规则列表中
                    previous_rules.extend(rule_lines)

    return previous_rules


def evaluate_rule_against_contradictions(rule: str, contradictions: List[str]) -> bool:
    """根据矛盾类型评估规则是否有效"""
    # 这里可以实现具体的矛盾类型评估逻辑
    # 假设我们通过LMM生成一个检查规则是否匹配矛盾类型的系统
    prompt = f"Check if the following rule matches any contradiction types (Factual, Content, Causal, Negation, Relational, Numeric, Temporal):\n{rule}"

    response = llm.generate(
        user_message=prompt,
        system_message="You are an expert in contradiction detection. Your task is to check if the given rule violates any contradiction types."
    )

    # 假设返回的响应是一个布尔值，表示是否匹配矛盾类型
    return "contradiction" in response.lower()  # 这里可以进一步调整根据实际的API响应进行调整

# 主程序入口
def main():
    # 加载历史规则
    previous_rules = load_previous_rules_from_folder(output_folder)

    # 遍历所有 JSON 文件
    json_files = [os.path.join(json_data_folder, f) for f in os.listdir(json_data_folder) if f.endswith(".json")]

    if not json_files:
        print(f"⚠️ 没有找到任何 JSON 文件在目录: {json_data_folder}")
        exit(1)

    # For each JSON file, generate predicates and rules and save to output
    for json_file in json_files:
        # Construct output file path based on the input JSON file
        output_file_path = os.path.join(output_folder, f"generated_rules_llm_{os.path.basename(json_file)}.txt")

        print(f"Processing file: {json_file}")

        # Call the function to generate predicates and rules for each JSON file, passing previous rules for learning
        generate_rico_predicates_and_rules_with_learning(
            graph_json_path=json_file,
            output_path=output_file_path,
            llm=llm,
            contradictions=contradictions,
            num_predicates=100,
            num_rules=100,
            previous_rules=previous_rules  # 将历史规则传递进去
        )

if __name__ == '__main__':
    main()

