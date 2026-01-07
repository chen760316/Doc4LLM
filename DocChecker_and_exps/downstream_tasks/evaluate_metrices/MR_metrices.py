"""
关注的核心是矛盾注入（冲突证据）如何影响模型对“原始答案”的信心**（即模型对原始答案的概率）**。
在这种情况下，我们不再关注模型最终选择哪个选项作为预测答案，而是直接分析模型赋予原始答案 (OA) 的概率变化。
我们仍然可以使用 OAR（原始答案比）和 CAR（反答案比）的概念，但它们的定义将从预测次数转变为概率比率，从而更好地衡量信心的变化。

💻 基于概率衡量知识冲突的指标实现为了衡量注入矛盾对模型坚持原有答案的信心的影响，我们需要比较模型在两种情况下的概率：
基线 (Baseline)：仅输入 QA对（无冲突证据）时，模型对 原始答案 (OA) 的概率 $P_{\text{Base}}(OA)$。冲突 (Conflict)：输入 QA对 + 冲突证据时，模型对 原始答案 (OA) 的概率 $P_{\text{Conflict}}(OA)$。
"""

from typing import List, Dict, Any, Tuple


# 假设的数据结构
class ProbabilitySample:
    def __init__(self, qa_id: str, prompt_base: str, prompt_conflict: str, original_answer_token: str, conflict_answer_token: str):
        self.qa_id = qa_id
        # 仅包含QA对的提示（用于基线）
        self.prompt_base = prompt_base
        # 包含QA对和冲突证据的提示
        self.prompt_conflict = prompt_conflict
        # 原始答案的标记/token (例如 'Paris', '2024')
        self.original_answer_token = original_answer_token
        # 冲突证据支持的答案的标记/token (例如 'London', '2023')
        self.conflict_answer_token = conflict_answer_token


# 假设的输入数据列表
PROBABILITY_DATASET: List[ProbabilitySample] = [
    # 示例数据
    ProbabilitySample("Q1", "The capital of France is...", "Conflict: The capital of France is London. The capital of France is...", 'Paris', 'London'),
    # ... 更多样本 ...
]


def get_token_probability(prompt: str, target_token: str) -> float:
    """
    【模型交互模拟函数】
    将Prompt输入给LLM，获取模型在下一个位置生成 'target_token' 的概率。

    在实际实现中，这需要调用模型的API或本地推理函数，并获取特定 token 的概率。

    返回: 目标 token 的概率 (0.0 到 1.0)
    """
    # 实际需替换为真实的模型接口调用
    # 例如：
    # probabilities = model.get_next_token_probabilities(prompt)
    # return probabilities.get(target_token, 0.0)
    return 0.5  # 占位符，假设概率为 0.5


def calculate_confidence_impact_metrics(dataset: List[ProbabilitySample]) -> Dict[str, float]:
    """
    计算基于概率的知识冲突指标，衡量冲突证据对原始答案信心的影响。
    """

    total_samples = len(dataset)
    if total_samples == 0:
        return {"Avg_Prob_Drop_OA": 0.0, "Avg_OAR_Prob": 0.0, "Avg_CAR_Prob": 0.0, "MR_Prob_Ratio": 0.0}

    # 累加器
    sum_prob_base_OA = 0.0  # Sum( P_Base(OA) )
    sum_prob_conflict_OA = 0.0  # Sum( P_Conflict(OA) )
    sum_prob_conflict_CA = 0.0  # Sum( P_Conflict(CA) )

    for sample in dataset:
        # --- 1. 获取基线概率 ---
        # 确保QA对是模型已知且能正确回答的，所以 P_Base(OA) 应该接近 1.0
        prob_base_OA = get_token_probability(sample.prompt_base, sample.original_answer_token)
        sum_prob_base_OA += prob_base_OA

        # --- 2. 获取冲突场景下的概率 ---

        # 冲突提示下，原始答案 (OA) 的概率
        prob_conflict_OA = get_token_probability(sample.prompt_conflict, sample.original_answer_token)
        sum_prob_conflict_OA += prob_conflict_OA

        # 冲突提示下，冲突答案 (CA) 的概率 (模型被说服的概率)
        prob_conflict_CA = get_token_probability(sample.prompt_conflict, sample.conflict_answer_token)
        sum_prob_conflict_CA += prob_conflict_CA

    # --- 3. 计算最终指标 ---

    # 1. 原始答案平均概率 (Avg_P_OA) - 衡量模型平均坚持信心的程度
    avg_P_OA = sum_prob_conflict_OA / total_samples

    # 2. 概率下降率 (Average Probability Drop of OA) - 衡量冲突的直接影响
    # (P_Base(OA) - P_Conflict(OA)) / P_Base(OA)
    avg_prob_base_OA = sum_prob_base_OA / total_samples

    if avg_prob_base_OA > 0:
        # 我们假设 P_Base(OA) 接近 1.0
        avg_prob_drop_OA = (avg_prob_base_OA - avg_P_OA) / avg_prob_base_OA
    else:
        avg_prob_drop_OA = 0.0

    # 3. 平均原始答案概率比 (OAR_Prob) - 相当于基于概率的 OAR
    # 衡量模型对原始答案的平均信心
    avg_OAR_Prob = avg_P_OA

    # 4. 平均冲突答案概率比 (CAR_Prob) - 相当于基于概率的 CAR
    # 衡量模型对冲突答案的平均采信程度
    avg_CAR_Prob = sum_prob_conflict_CA / total_samples

    # 5. 记忆比率 (MR) - 基于概率（Probability-based Memorization Ratio）
    # MR = Avg_P_OA / (Avg_P_OA + Avg_P_CA)

    denominator = avg_OAR_Prob + avg_CAR_Prob
    MR_Prob_Ratio = avg_OAR_Prob / denominator if denominator > 0 else 0.0

    return {
        # 核心指标：原始答案的平均概率下降
        "Avg_Prob_Drop_OA": avg_prob_drop_OA,

        # 概率形式的 OAR 和 CAR
        "Avg_OAR_Prob (Model_Confidence_on_OA)": avg_OAR_Prob,
        "Avg_CAR_Prob (Model_Trust_on_Conflict)": avg_CAR_Prob,

        # 基于概率比率的记忆比率
        "MR_Prob_Ratio": MR_Prob_Ratio,

        "Total_Samples": total_samples
    }

# --- 运行示例（需要实现 get_token_probability 函数） ---
results = calculate_confidence_impact_metrics(PROBABILITY_DATASET)
print(results)