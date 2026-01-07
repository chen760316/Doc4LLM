"""
匹配规则中的谓词和，矛盾句子对的关键实体，关系，属性，两者重合度高认为检测到了相应矛盾句子对
无需将规则转换为自然语言
"""
import json
import re
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import os

# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=8000
)

# 获取矛盾类型函数
def get_contradiction_type(s1, s2):
    rules_prompt = f"""
    You are an expert in detecting contradiction types between two sentences.

    Sentence 1: {s1}
    Sentence 2: {s2}

    Please classify the contradiction type between these two sentences.
    Provide only the contradiction type, no explanations. 
    Possible contradiction types: 'Numeric', 'Temporal', 'Factual', 'Emotional', 'Relation', etc.
    """

    system_message = "You are a helpful assistant that analyzes contradictions between sentences."

    # 调用 LLM 进行判断
    response = llm.generate(user_message=rules_prompt, system_message=system_message)

    # 解析返回的响应，提取矛盾类型部分
    contradiction_type = response.strip()  # 直接返回模型的响应

    return contradiction_type


# 读取矛盾数据
def load_contradictions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# 读取规则文件
def load_rules(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


# 提取规则中的实体、属性和关系
def extract_entities_attributes_from_rule(rule):
    """
    提取规则中的实体、属性和关系（简化版）。假设规则以某种固定格式给出。
    可以根据具体规则结构进行调整。
    """
    # 使用正则表达式从规则中提取实体、属性、关系等部分
    entities = re.findall(r'\b[A-Z][a-z]*\b', rule)  # 假设实体首字母大写
    attributes = re.findall(r'\b\w+\b', rule)  # 假设属性是由单词组成的
    return entities, attributes


# 检查规则是否覆盖矛盾类型
def is_rule_covering_contradiction(rule, entities_s1, entities_s2, attributes_s1, attributes_s2, threshold=0.5):
    """
    判断规则是否覆盖了矛盾的主体（即矛盾涉及的实体和属性），
    不要求完全匹配，只需要大部分覆盖。
    rule 是自然语言规则，entities_s1 和 entities_s2 是从矛盾句子中提取的实体和属性。
    attributes_s1 和 attributes_s2 是对应实体的属性。
    threshold 是匹配覆盖的比例阈值，默认是 50%。
    """
    # 合并句子中的实体和属性
    all_entities = entities_s1 + entities_s2
    all_attributes = attributes_s1 + attributes_s2

    # 计算总的实体和属性数量
    total_entities = len(all_entities)
    total_attributes = len(all_attributes)

    # 计算规则匹配的实体和属性数量
    matched_entities = 0
    matched_attributes = 0

    # 遍历实体，检查规则是否包含实体
    for entity in all_entities:
        if entity.lower() in rule.lower():  # 实体匹配
            matched_entities += 1

    # 遍历属性，检查规则是否包含属性
    for attribute in all_attributes:
        if attribute.lower() in rule.lower():  # 属性匹配
            matched_attributes += 1

    # 计算覆盖比例
    entity_coverage = matched_entities / total_entities if total_entities > 0 else 0
    attribute_coverage = matched_attributes / total_attributes if total_attributes > 0 else 0

    # 如果实体和属性的匹配覆盖度超过了阈值，就认为规则覆盖了矛盾
    if entity_coverage + attribute_coverage >= threshold:
        return True

    return False



# 检查规则是否覆盖矛盾类型
def check_rule_coverage(contradictions, rules):
    covered_types = set()
    uncovered_types = set()
    covered_count = 0  # 被覆盖的矛盾数
    total_count = len(contradictions)  # 总的矛盾数

    # 用于记录哪些具体矛盾被哪些规则覆盖
    covered_contradictions = []  # 存储被规则覆盖的具体矛盾
    uncovered_contradictions = []  # 存储未被规则覆盖的具体矛盾

    for contradiction in contradictions:
        s1 = contradiction['sentence_1']
        s2 = contradiction['sentence_2']

        # 获取 LLM 返回的矛盾类型
        contradiction_type = get_contradiction_type(s1, s2)

        # 提取句子中的实体和属性
        entities_and_attributes = extract_entities_and_attributes_from_llm(s1, s2)

        # 获取句子1和句子的实体与属性
        entities_s1 = entities_and_attributes["entities_s1"]
        attributes_s1 = entities_and_attributes["attributes_s1"]

        # 获取句子2和句子的实体与属性
        entities_s2 = entities_and_attributes["entities_s2"]
        attributes_s2 = entities_and_attributes["attributes_s2"]

        # 检查规则文件是否已有相关规则
        rule_covered = False
        for idx, rule in enumerate(rules):
            if is_rule_covering_contradiction(rule, entities_s1, entities_s2, attributes_s1, attributes_s2):
                rule_covered = True
                covered_count += 1
                covered_contradictions.append({
                    'contradiction': f"Sentence 1: {s1} / Sentence 2: {s2}",
                    'rule': rule,  # 存储规则
                    'original_rule': rules[idx]  # 输出原始规则（即未转换前的规则）
                })
                # 记录被覆盖的矛盾类型
                covered_types.add(contradiction_type)
                break  # 一旦发现有规则覆盖该矛盾句子对，跳出规则遍历

        if not rule_covered:
            uncovered_types.add(contradiction_type)
            uncovered_contradictions.append(f"Sentence 1: {s1} / Sentence 2: {s2}")

    # 输出总结
    print("\nSummary of coverage:")
    print(f"Total contradictions: {total_count}")
    print(f"Covered contradictions: {covered_count}")
    print(f"Coverage percentage: {(covered_count / total_count) * 100:.2f}%")

    # 输出具体覆盖的矛盾和规则
    print("\nCovered contradictions and corresponding rules:")
    for contradiction in covered_contradictions:
        print(f"Contradiction: {contradiction['contradiction']}")
        print(f"Rule (Natural Language): {contradiction['rule']}")
        print(f"Original Rule: {contradiction['original_rule']}")  # 输出原始规则
        print("-" * 80)

    # 输出未覆盖的矛盾
    print("\nUncovered contradictions:")
    for contradiction in uncovered_contradictions:
        print(contradiction)

    print(f"\nCovered types: {', '.join(covered_types)}")
    print(f"Uncovered types: {', '.join(uncovered_types)}")



def extract_entities_and_attributes_from_llm(s1, s2):
    """
    使用 LLM 从两个句子中提取涉及的实体及其属性。

    参数:
    - s1 (str): 第一个句子。
    - s2 (str): 第二个句子。

    返回:
    - dict: 包含 "entities_s1", "entities_s2", "attributes_s1", "attributes_s2" 的字典。
    """
    # 构建提示，要求 LLM 提取两个句子中的实体及属性
    prompt = f"""
    Please analyze the following two sentences and extract the entities (e.g., persons, locations, objects, etc.) and their attributes (e.g., actions, relations, descriptors).

    Sentence 1: "{s1}"
    Sentence 2: "{s2}"

    For each sentence, please return the following:
    - Entities: [list of entities in sentence]
    - Attributes: [list of attributes associated with those entities]

    Output the results as follows:
    Sentence 1:
    Entities: [list of entities]
    Attributes: [list of attributes]

    Sentence 2:
    Entities: [list of entities]
    Attributes: [list of attributes]
    """

    # 调用 LLM 进行实体和属性提取
    response = llm.generate(
        user_message=prompt,
        system_message="You are an expert at analyzing sentences and extracting entities and attributes."
    )

    try:
        # 解析模型返回的内容，提取实体和属性
        # 假设返回内容格式类似：
        # Sentence 1:
        # Entities: ["John", "Berlin"]
        # Attributes: ["delivered a keynote speech", "in 2021"]
        # Sentence 2:
        # Entities: ["Tom", "San Francisco"]
        # Attributes: ["gave a lecture", "in 2022"]

        # 分割句子1和句子2的结果
        sentence_1_section = response.split("Sentence 1:")[1].split("Sentence 2:")[0].strip()
        sentence_2_section = response.split("Sentence 2:")[1].strip()

        # 提取句子1的实体和属性
        entities_s1 = [entity.strip() for entity in sentence_1_section.split("Entities:")[1].split("Attributes:")[0].strip().strip('[]').split(',')]
        attributes_s1 = [attribute.strip() for attribute in sentence_1_section.split("Attributes:")[1].strip().strip('[]').split(',')]

        # 提取句子2的实体和属性
        entities_s2 = [entity.strip() for entity in sentence_2_section.split("Entities:")[1].split("Attributes:")[0].strip().strip('[]').split(',')]
        attributes_s2 = [attribute.strip() for attribute in sentence_2_section.split("Attributes:")[1].strip().strip('[]').split(',')]

        # 返回提取的实体和属性
        return {
            "entities_s1": entities_s1,
            "attributes_s1": attributes_s1,
            "entities_s2": entities_s2,
            "attributes_s2": attributes_s2
        }

    except Exception as e:
        print(f"Error in entity extraction: {e}")
        return {
            "entities_s1": [],
            "attributes_s1": [],
            "entities_s2": [],
            "attributes_s2": []
        }

# 执行矛盾分析
def analyze_coverage(contradiction_file, rule_file):
    contradictions = load_contradictions(contradiction_file)
    rules = load_rules(rule_file)
    check_rule_coverage(contradictions, rules)


# 分析矛盾覆盖情况
analyze_coverage('dense_text_cross.txt', 'rules.txt')
