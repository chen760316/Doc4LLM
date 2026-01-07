"""
弃用
基于NLI的模型（判断语义冲突或蕴含）
RoBERTa-Large (MNLI)具有强大的句子对建模能力，判断冲突性/矛盾性，映射为 [0,1] 评分
不准确
这个模型是在 MNLI（Multi-Genre Natural Language Inference） 数据集上训练的，它的特点是：
特点	描述
✅ 强项	判断句法、语义层面上的蕴含/中立/矛盾（不依赖复杂推理）
❌ 弱项	缺乏常识、时间逻辑、否定推理能力
useless（仅写作）
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 载入MNLI任务上的模型
tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")


def compute_conflict_score(ex_a, ey_b):
    # ex_a, ey_b 是自然语言表示的事件，例如：
    # ex_a = "Tom was born in 1990"
    # ey_b = "Tom died in 1980"

    inputs = tokenizer(ex_a, ey_b, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)

    # MNLI的标签顺序通常是 [entailment, neutral, contradiction]
    contradiction_score = probs[0][2].item()
    return contradiction_score

def classify_conflict(score, threshold_high=0.7, threshold_low=0.3):
    if score > threshold_high:
        return "❌ 强冲突"
    elif score < threshold_low:
        return "✅ 不冲突/蕴含"
    else:
        return "⚠️ 中立/可能冲突"

# 测试句子对
test_cases = [
    ("The man is eating.", "The man is not eating."),
    ("The cat is sleeping on the couch.", "There is a cat on the sofa."),
    ("Tom was born in 1990", "Tom died in 2020"),
    ("Tom was born in 1990", "Tom died in 1980"),
]

# 执行测试并打印结果
for premise, hypothesis in test_cases:
    score = compute_conflict_score(premise, hypothesis)
    label = classify_conflict(score)
    print(f"前提: \"{premise}\"\n假设: \"{hypothesis}\"\n冲突得分: {score:.4f} → {label}\n")