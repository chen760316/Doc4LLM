"""
谓词M(𝑥. ¯𝐴,𝑦. ¯𝐵) | M(𝑒𝑥. ¯𝐴,𝑒𝑦. ¯𝐵)的实现

用于推理时间顺序
借助大模型实现
"""

"""
基于NLU语义匹配的模型：判断上下文依赖或共现
借助llm实现
"""

from knowledge_graph_maker import OpenAIClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize your LLM client
llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

def infer_temporal_relation_openai(event_a: str, event_b: str) -> tuple[str, float]:
    """
    Use OpenAI LLM to infer temporal order between two events.
    Returns: (answer string, score between 0 and 1)
    """
    system_message = (
        "You are an assistant that determines the temporal relationship between two events. "
        "Your task is to answer whether event A happened before event B. "
        "Answer with only one word: 'yes', 'no', or 'unknown'."
    )

    user_message = f"""
Given two events:
A: {event_a}
B: {event_b}

Question: Did event A happen before event B?
Answer with only one word: yes, no, or unknown.
"""

    # Call the OpenAI model
    response = llm.generate(system_message, user_message)
    print("Raw LLM response:", response)

    # Parse and normalize the response
    answer = response.strip().lower()
    score_map = {
        "yes": 1.0,
        "no": 0.0,
        "unknown": 0.5
    }
    score = score_map.get(answer, 0.5)  # fallback to 0.5 if unrecognized

    return answer, score

def get_later_event(event_a: str, event_b: str):
    """
    判断哪个事件在时序上更靠后
    """
    ans, score = infer_temporal_relation_openai(event_a, event_b)

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
if __name__ == "__main__":
    a = "Tom graduated from college in 2010"
    b = "Tom got his first job at Google in 2012"

    ans, score = get_later_event(a, b)
    print(f"Prediction: {ans}, Score: {score}")

