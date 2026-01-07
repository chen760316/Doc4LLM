# 📌 1. 外部依赖和辅助函数（假设）
# 我们需要一个能够计算 BERTScore 的函数，以及一个用于处理数据的结构。

# 假设的外部库/函数
from typing import List, Tuple, Dict
# from bert_score import score as bert_score_calc # 实际中需要导入BERTScore库

def bert_score_calc(hypotheses: List[str], references: List[str]) -> Tuple[List[float], List[float]]:
    """
    一个计算 BERTScore P/R 的占位函数。
    返回 (Precision列表, Recall列表)
    """
    # 实际实现中，这里会调用 BERTScore 库
    # 为简化，我们返回一个占位的高分
    P = [0.99] * len(hypotheses) # 假设得分很高
    R = [0.99] * len(hypotheses)
    return P, R

# 假设的数据结构
class Sample:
    def __init__(self, gt_label: str, gt_evidence: List[str], pred_label: str, pred_evidence: List[str]):
        # 基本事实 (Ground Truth)
        self.gt_label = gt_label         # 'Yes' (矛盾) 或 'No' (无矛盾)
        self.gt_evidence = gt_evidence   # GT的证据句子列表
        # 模型预测 (Prediction)
        self.pred_label = pred_label     # 'Yes' 或 'No'
        self.pred_evidence = pred_evidence # 模型预测的证据句子列表

# 假设的输入数据列表
DATASET: List[Sample] = [
    # 示例数据（您需要替换为您的实际数据）
    Sample('Yes', ['GT evidence 1'], 'Yes', ['Predicted evidence 1']), # TP 示例
    Sample('Yes', ['GT evidence 2'], 'No', ['']),                     # FN 示例
    Sample('No', [''], 'Yes', ['Mistake evidence']),                  # FP 示例
    Sample('No', [''], 'No', ['']),                                   # TN 示例
    # ... 更多样本 ...
]

# 📌 2. 核心逻辑：证据验证 $V(E)$ 函数这是根据您的严格要求实现的函数，
# 即 $\text{BERTScore}_{\text{Precision}} > 0.98$ 或 $\text{BERTScore}_{\text{Recall}} > 0.98$。

def verify_evidence_V_E(pred_evidence: List[str], gt_evidence: List[str]) -> bool:
    """
    评价证据是否被正确识别 V(E)
    只有当任何选定的证据句子的BERTScore准确率或召回率高于0.98时，V(E)才为True。
    """
    if not gt_evidence or not pred_evidence:
        # 如果是'No'的样本，或者没有证据，V(E)的判断逻辑应该根据上下文调整。
        # 在'Yes'的判定中，如果任何一个为空，通常是 False。
        return False

    # 我们需要比较每个预测证据与所有GT证据
    for p_sent in pred_evidence:
        # 计算当前预测证据与所有基本事实证据的 BERTScore
        # 假设我们只取当前 p_sent 与所有 gt_evidence 中得分最高的那个。
        # 注意：BERTScore 函数通常一次性处理多个句子。

        # 为了简化和性能，这里假设只计算一次，并在内部找到最佳匹配
        P_scores, R_scores = bert_score_calc([p_sent] * len(gt_evidence), gt_evidence)

        # 查找当前预测句子与所有 GT 证据的最佳匹配得分
        max_P = max(P_scores)
        max_R = max(R_scores)

        # 判断是否语义等同
        if max_P > 0.98 or max_R > 0.98:
            return True  # 只要有一个预测证据满足条件，V(E)即为 True

    return False  # 所有预测证据都未能达到语义等同标准

# 📌 3. 主函数：指标计算

def calculate_contradiction_metrics(dataset: List[Sample]) -> Dict[str, float]:
    """
    计算二元文档矛盾检测任务的复合指标。
    """
    # 计数器初始化
    TP = 0  # True Positive (J(D)=True 且 V(E)=True)
    TN = 0  # True Negative (J(D)=False)
    FP = 0  # False Positive
    FN = 0  # False Negative

    # 用于EHR指标：成功识别正确证据的样本计数 (Success_Evidence_Hits)
    # 在这个复合标准下，Success_Evidence_Hits 实际上就是 TP

    for sample in dataset:
        gt_is_yes = (sample.gt_label == 'Yes')  # 实际有矛盾
        pred_is_yes = (sample.pred_label == 'Yes')  # 预测有矛盾

        # --- A. 复合正确性判定：J(D) ∧ V(E) ---

        is_V_E_correct = False
        if pred_is_yes:
            # 只有当模型预测为 'Yes' 时，才需要验证证据 V(E)
            is_V_E_correct = verify_evidence_V_E(sample.pred_evidence, sample.gt_evidence)

        # --- B. 计算混淆矩阵元素 ---

        if gt_is_yes and pred_is_yes:
            # 实际 Yes，预测 Yes
            # 只有当证据 V(E) 也正确时，才计为 TP
            if is_V_E_correct:
                TP += 1
            else:
                # 预测标签和GT标签都为 'Yes'，但证据不合格，计为 FN
                # 这是严格标准下的关键点：预测 'Yes' 但 V(E) 不合格，结果被视为错误 (FN)
                FN += 1

        elif gt_is_yes and not pred_is_yes:
            # 实际 Yes，预测 No (漏报)
            FN += 1

        elif not gt_is_yes and pred_is_yes:
            # 实际 No，预测 Yes (误报)
            # 无需考虑 V(E)，因为 J(D) 本身就错了
            FP += 1

        elif not gt_is_yes and not pred_is_yes:
            # 实际 No，预测 No (正确拒绝)
            TN += 1

    # --- C. 计算最终指标 ---

    total = TP + TN + FP + FN
    P_GT = TP + FN  # 实际正类总数
    P_Pred = TP + FP  # 预测正类总数

    # 1. 准确性 (Accuracy)
    accuracy = (TP + TN) / total if total > 0 else 0.0

    # 2. 精确度 (Precision) - 基于复合TP
    precision = TP / P_Pred if P_Pred > 0 else 0.0

    # 3. 召回率 (Recall) - 基于复合TP
    recall = TP / P_GT if P_GT > 0 else 0.0

    # 4. F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 5. EHR 指标 (EHR)
    # EHR 表示成功识别正确证据的样本比例，假设模型已预测为“是”，且实际为“是”
    # 在这个复合标准下，TP (复合) = 成功识别正确证据的样本数
    ehr = recall  # EHR = TP / (TP + FN) = Recall

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "EHR_Metric": ehr,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN
    }

# --- 运行示例 ---
results = calculate_contradiction_metrics(DATASET)
print(results)