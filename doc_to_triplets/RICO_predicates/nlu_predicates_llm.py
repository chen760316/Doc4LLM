"""
谓词M(𝑥. ¯𝐴,𝑦. ¯𝐵) | M(𝑒𝑥. ¯𝐴,𝑒𝑦. ¯𝐵)的实现
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

def judge_dependency_and_cooccurrence(sentence_a: str, sentence_b: str) -> tuple[float, float, str]:
    """
    Use LLM to judge contextual dependency and co-occurrence between two sentences.
    Returns (dependency_score, cooccurrence_score, decision).
    """

    system_message = (
        "You are an assistant that evaluates two sentences for:\n"
        "1) Contextual dependency — how much the meaning of one sentence depends on the other.\n"
        "2) Co-occurrence — how often these two sentences/events appear together in the same context.\n"
        "Return two decimal numbers between 0 and 1 separated by a comma, "
        "first is dependency score, second is co-occurrence score."
    )

    user_message = f"""
Sentence A: "{sentence_a}"
Sentence B: "{sentence_b}"

Please respond ONLY with two decimal numbers separated by a comma, e.g. "0.45, 0.78"
"""

    response = llm.generate(system_message, user_message)
    print("Raw LLM response:", response)

    try:
        dep_str, cooc_str = response.strip().split(",")
        dep_score = float(dep_str.strip())
        cooc_score = float(cooc_str.strip())

        dep_score = max(0.0, min(1.0, dep_score))
        cooc_score = max(0.0, min(1.0, cooc_score))
    except Exception as e:
        print("Failed to parse dependency and co-occurrence scores:", e)
        dep_score, cooc_score = 0.0, 0.0

    # Decision logic based on thresholds
    dependency_threshold = 0.8
    cooccurrence_threshold = 0.6

    if dep_score >= dependency_threshold:
        decision = "dependency"
    elif cooc_score >= cooccurrence_threshold:
        decision = "co-occurrence"
    else:
        decision = "neither"

    return dep_score, cooc_score, decision


if __name__ == "__main__":
    a = "Tom was born in 1990."
    b = "Tom graduated from university in 2012."

    dep, cooc, decision = judge_dependency_and_cooccurrence(a, b)
    print(f"Dependency score: {dep:.4f}")
    print(f"Co-occurrence score: {cooc:.4f}")
    print(f"Decision: {decision}")


