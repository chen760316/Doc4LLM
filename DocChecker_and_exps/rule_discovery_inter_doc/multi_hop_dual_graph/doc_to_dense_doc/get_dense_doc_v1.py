"""
采用了句子分割模型把原始文本text1.txt和他text2.txt拆分为了句子对
执行效率低
useless
"""
import os
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import nltk

# 下载句子分割模型（首次运行需要）
nltk.download('punkt')

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 初始化 LLM
llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# 载入 system prompt（与原代码相同）
system_prompt = """
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
"""

def detect_contradiction(pair, llm):
    """判断两句是否矛盾"""
    prompt = f"""
Sentence 1: "{pair['sentence_1']}"
Sentence 2: "{pair['sentence_2']}"
Do these sentences contradict each other? Answer only 'Yes' or 'No'.
"""
    sys_prompt = "You are a language reasoning assistant. Only answer 'Yes' if the sentences are contradictory, otherwise 'No'."
    resp = llm.generate(user_message=prompt, system_message=sys_prompt)
    return resp.strip().lower() == 'yes'


def classify_contradiction_type(pair, llm):
    """分类矛盾类型"""
    prompt = f"""
Sentence 1: "{pair['sentence_1']}"
Sentence 2: "{pair['sentence_2']}"
If these sentences contradict, what type of contradiction is it? Explain briefly.
"""
    resp = llm.generate(user_message=prompt, system_message=system_prompt)
    return resp.strip()


def extract_contradictions_from_text(file_path, llm):
    """从单个文件中提取出所有句子之间的矛盾对"""
    from nltk.tokenize import sent_tokenize

    # 1. 读取并分句
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = sent_tokenize(text)

    contradictions = []
    n = len(sentences)

    # 2. 句子两两对比（可优化为滑动窗口）
    for i in range(n):
        for j in range(i + 1, n):
            pair = {"sentence_1": sentences[i], "sentence_2": sentences[j]}
            if detect_contradiction(pair, llm):
                contradiction_type = classify_contradiction_type(pair, llm)
                contradictions.append({
                    "s1": sentences[i],
                    "s2": sentences[j],
                    "type": contradiction_type
                })
                print(f"✅ Found contradiction ({contradiction_type}) between:\n - {sentences[i]}\n - {sentences[j]}\n")

    return contradictions


def save_dense_text(contradictions, output_path):
    """将矛盾句对写入 dense 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for c in contradictions:
            s1 = c['s1']
            s2 = c['s2']
            ctype = c['type']
            f.write(f"[{ctype}]\n{s1}\n{s2}\n\n")


def main():
    # 对 text1.txt 和 text2.txt 分别执行
    for i in [1, 2]:
        input_file = f"text{i}.txt"
        output_file = f"dense_text{i}.txt"

        print(f"\n=== Processing {input_file} ===")
        contradictions = extract_contradictions_from_text(input_file, llm)
        save_dense_text(contradictions, output_file)
        print(f"💾 Saved dense contradiction subset to {output_file}\n")


if __name__ == "__main__":
    main()
