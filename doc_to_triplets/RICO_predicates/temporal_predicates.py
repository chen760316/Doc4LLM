"""
谓词M(𝑥. ¯𝐴,𝑦. ¯𝐵) | M(𝑒𝑥. ¯𝐴,𝑒𝑦. ¯𝐵)的实现

用于推理时间顺序/上下文依赖的模型
方法一：T5 / Flan-T5 Prompt-based 推理
这种方式更灵活且简单，适合通用事件时间顺序判断
可用，结果判断准确
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 使用 Flan-T5 模型（轻量版）
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def infer_temporal_relation(event_a: str, event_b: str):
    """
    使用 prompt 推理事件时间先后
    """
    prompt = f"""
    Given two events:
    A: {event_a}
    B: {event_b}
    Question: Did event A happen before event B? Answer with "yes", "no", or "unknown".
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=5)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    # 转为可用分数
    score = {
        "yes": 1.0,
        "no": 0.0,
        "unknown": 0.5
    }.get(answer, 0.5)

    return answer, score


def get_later_event(event_a: str, event_b: str):
    """
    判断哪个事件在时序上更靠后
    """
    ans, score = infer_temporal_relation(event_a, event_b)

    if ans == "yes":
        # A happens before B, so B is later
        return "Event B happens later", score
    elif ans == "no":
        # A happens after B, so A is later
        return "Event A happens later", score
    else:
        # unknown, no clear answer
        return "The temporal order is unclear", score


# 示例
a = "Tom graduated from college in 2010"
b = "Tom got his first job at Google in 2012"

ans, score = get_later_event(a, b)
print(f"Prediction: {ans}, Score: {score}")
