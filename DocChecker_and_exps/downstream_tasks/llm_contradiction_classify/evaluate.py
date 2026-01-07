"""
使用的数据集：knowledge conflict
使用的基准："gpt-4.1-mini"
"""
import json
import os
import random
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# 假设 knowledge_graph_maker.py 中定义了 OpenAIClient
# 请确保您的 OpenAIClient 类包含一个 `generate(self, user_message: str, system_message: str = None)` 方法
try:
    from knowledge_graph_maker import OpenAIClient
except ImportError:
    print("⚠️ 导入 OpenAIClient 失败。请确保 'knowledge_graph_maker' 文件存在且 OpenAIClient 类已定义。")


    # 临时定义一个模拟类以便代码结构完整
    class OpenAIClient:
        def __init__(self, model: str, **kwargs):
            print(f"--- 警告：使用模拟 LLM 客户端：{model} ---")
            self.model = model

        def generate(self, user_message: str, system_message: str = None) -> str:
            # 模拟 LLM 响应，用于结构测试
            time.sleep(0.5)
            if "Conflict" in user_message:
                return "The answer based on external knowledge is: [Simulated Conflicting Answer]"
            return "The answer is: [Simulated Parametric Answer]"


    import time

# --- 1. 配置和 LLM 客户端 ---
load_dotenv()
# 假设 API 密钥和 Base URL 已通过 .env 加载

# 初始化 LLM 客户端
try:
    # 建议使用性能更高的模型进行事实核查
    llm = OpenAIClient(
        model="gpt-4.1-mini",
        temperature=0.0,  # 事实核查应使用较低温度
        max_tokens=256
    )
except Exception as e:
    print(f"Error initializing OpenAIClient: {e}. 无法进行 LLM 实验。")
    llm = None
    exit()  # 如果 LLM 无法初始化，则退出

# --- 2. 数据加载 ---

DATA_FILE = '../../../datasets/data_from_related_work/DSCD_datasets/knowledge conflict/data.json'
OUTPUT_FILE = 'llm_conflict_results.json'


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 data.json 数据集，并确保返回的是数据点的列表。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 核心修正：检查顶级结构是否为字典
        if isinstance(raw_data, dict):
            # 如果是字典，则提取所有值并转换为列表
            data_list = list(raw_data.values())
        elif isinstance(raw_data, list):
            # 如果已经是列表，直接使用
            data_list = raw_data
        else:
            print(f"❌ 错误：数据集顶级结构不是字典或列表，而是 {type(raw_data)}。")
            return []

        print(f"✅ 成功加载 {len(data_list)} 条数据。")
        return data_list
    except FileNotFoundError:
        print(f"❌ 错误：找不到数据集文件 {file_path}。请确保文件存在。")
        return []
    except json.JSONDecodeError:
        print(f"❌ 错误：文件 {file_path} 不是有效的 JSON 格式。")
        return []


# --- 3. Prompt Template 定义 (对应研究场景) ---

# 基础 System Prompt
BASE_SYSTEM_PROMPT = (
    "You are a factual reasoning agent. Your task is to answer the user's question based STRICTLY "
    "on the external evidence provided. Do NOT use your parametric knowledge."
)


def create_prompt_S1(data_point: Dict[str, Any]) -> str:
    """场景 1：仅提供冲突知识。"""
    # 使用 question_about_conflicting_segments 提问
    return (
        f"External Evidence:\n---\n{data_point['conflicting_knowledge']}\n---\n"
        f"Question: {data_point['question_about_conflicting_segments']}\n"
        "Answer based only on the External Evidence."
    )


def create_prompt_S2(data_point: Dict[str, Any]) -> str:
    """场景 2：提供冲突知识 + 参数知识（作为支持证据）。"""
    # 随机化证据顺序，避免位置偏差
    evidence_list = [
        {"type": "Conflicting", "text": data_point['conflicting_knowledge']},
        {"type": "Parametric", "text": data_point['parametric_knowledge']}
    ]
    random.shuffle(evidence_list)

    evidence_str = "\n---\n".join([f"Evidence {i + 1} ({item['type']}): {item['text']}" for i, item in enumerate(evidence_list)])

    return (
        f"External Evidence (Note: Evidence may conflict with each other or your internal knowledge):\n{evidence_str}\n"
        f"Question: {data_point['question_about_conflicting_segments']}\n"
        "Answer based only on the provided External Evidence."
    )


def create_enhanced_prompt_S4(data_point: Dict[str, Any], prompt_type: str) -> Tuple[str, str]:
    """场景 4：增强的 Prompting (针对场景 1 或 2)。"""

    # 增强描述：根据冲突类型构建更详细的 Context
    # 注意：数据集缺少时间/语义冲突标签，我们基于 substitution/shuffling 进行假设分类和增强。

    # 假设：
    # 语义冲突增强: 详细描述 main_entity
    # 时间冲突增强: (此数据集无明确时间，因此我们专注于语义增强)

    enhanced_context = f"The main topic for this question is: {data_point['main_entity']} ({data_point['parametric_knowledge'].split('.')[0]})."

    if prompt_type == 'S1':
        # 增强版场景 1
        user_prompt = create_prompt_S1(data_point)
        enhanced_user_prompt = f"{enhanced_context}\n\n{user_prompt}"
    else:
        # 增强版场景 2
        user_prompt = create_prompt_S2(data_point)
        enhanced_user_prompt = f"{enhanced_context}\n\n{user_prompt}"

    # 构建更详细的 System Prompt
    enhanced_system_prompt = (
        f"{BASE_SYSTEM_PROMPT} Ensure your answer is STRICTLY faithful to the source text and specifically addresses the conflict area."
        f"The conflict involves the entity '{data_point['entity_before']}' being incorrectly stated as '{data_point['entity_after']}'."
    )

    return enhanced_user_prompt, enhanced_system_prompt


# --- 4. 实验运行和结果记录 ---

def run_experiment(data: List[Any]):  # 将类型提示改为 List[Any] 以适应复杂结构
    """运行所有评估场景。"""

    results = []

    # 随机抽取 20 条数据进行演示
    sample_data = random.sample(data, min(20, len(data)))

    print(f"\n🧠 开始 LLM 事实一致性评估 (共 {len(sample_data)} 个样本)...")

    for i, item in enumerate(sample_data):  # item 现在是列表

        # 核心修正：解包 item 列表，获取真正的字典 data_point
        if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
            data_point = item[0]  # 取列表中的第一个字典作为数据点
        elif isinstance(item, dict):
            # 如果 item 已经是字典 (以防结构变异)，则直接使用
            data_point = item
        else:
            print(f"致命错误：样本 {i + 1} 结构异常。类型是 {type(item)}，无法解包。")
            continue  # 跳过当前错误样本

        # 再次检查 data_point 是否为字典，以确保安全访问
        if not isinstance(data_point, dict):
            print(f"致命错误：解包后样本 {i + 1} 仍然不是字典。请检查 data.json 结构。")
            continue

        print(f"\n--- 样本 {i + 1}/{len(sample_data)}: {data_point['main_entity']} ---")

        result_entry = data_point.copy()
        result_entry['llm_results'] = {}

        # ... (S1, S2, S4 场景的逻辑不变) ...
        # Scenario 1: External Conflict Only (仅冲突证据)
        prompt_s1 = create_prompt_S1(data_point)
        try:
            response_s1 = llm.generate(user_message=prompt_s1, system_message=BASE_SYSTEM_PROMPT)
            result_entry['llm_results']['S1_Conflict_Only'] = response_s1
            print(f"  S1 Response: {response_s1[:50]}...")
        except Exception as e:
            result_entry['llm_results']['S1_Conflict_Only'] = f"Error: {e}"

        # Scenario 2: Two Conflicts (冲突证据 + 参数证据)
        prompt_s2 = create_prompt_S2(data_point)
        try:
            response_s2 = llm.generate(user_message=prompt_s2, system_message=BASE_SYSTEM_PROMPT)
            result_entry['llm_results']['S2_Two_Conflicts'] = response_s2
            print(f"  S2 Response: {response_s2[:50]}...")
        except Exception as e:
            result_entry['llm_results']['S2_Two_Conflicts'] = f"Error: {e}"

        # Scenario 3: Enhanced Prompting (增强提示) - 基于 S2
        prompt_s4_user, prompt_s4_system = create_enhanced_prompt_S4(data_point, 'S2')
        try:
            response_s4 = llm.generate(user_message=prompt_s4_user, system_message=prompt_s4_system)
            result_entry['llm_results']['S4_Enhanced_Prompting'] = response_s4
            print(f"  S4 Response: {response_s4[:50]}...")
        except Exception as e:
            result_entry['llm_results']['S4_Enhanced_Prompting'] = f"Error: {e}"

        results.append(result_entry)

    return results


def save_results(results: List[Dict[str, Any]]):
    """保存实验结果。"""
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 实验结果已保存到 {OUTPUT_FILE}。")


# --- 5. 主执行逻辑 ---

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"请将您的数据集文件命名为 '{DATA_FILE}' 并放在当前目录下。")
    else:
        dataset = load_data(DATA_FILE)
        if dataset:
            # 运行实验
            final_results = run_experiment(dataset)

            # 保存结果
            save_results(final_results)

            # 简单评估指导
            print("\n--- 评估指导 ---")
            print("要评估 LLM 的忠实度，请手动/自动检查 llm_results 中的回答：")
            print("1. S1/S2 场景中，LLM 是否使用了冲突知识 (entity_after) 来回答 question_about_conflicting_segments？")
            print("2. S4 场景中，详细描述和增强的系统提示是否提高了 LLM 忠实于外部证据 (entity_after) 的比例？")
            print("理想情况下，当外部证据明确时 (S1)，LLM 应该回答冲突知识。")
            print("当证据冲突时 (S2)，LLM 倾向于其参数知识 (即回答 entity_before)，而 S4 的目标就是减少这种倾向。")