"""
将text.txt完整输入进大模型
尽可能找到text1.txt和text2.txt之间存在的各种矛盾类型
将结果保存进dense_text1.txt，dense_text2.txt和dense_text_cross.txt
"""
import os
import json
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv

# -------------------------
# 环境变量加载
# -------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# -------------------------
# 初始化 LLM
# -------------------------
llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.2,
    top_p=0.9,
    max_tokens=4000
)

# -------------------------
# 系统提示词（加入示例）
# -------------------------
system_prompt = """
You are an expert in analyzing documents to detect contradictions between two texts.

Your task is to read two passages and identify all pairs of statements from each text that contradict each other, classifying each contradiction into one of the following types: Factual, Content (Opinion/Emotion, Misinformation, Semantic/Lexical conflicts), Causal, Negation, Relation, Numeric, Temporal.

### Examples:
1. **Factual**:
   - Sentence 1: "The Nobel Prize was indeed established in 1901 to commemorate Alfred Nobel."
   - Sentence 2: "Hinton received a world-renowned scientific award in 2024, which was established in 1901 to honor Shannon's contributions to the field of information."
   Contradiction type: **Factual**
   Reason: The Nobel Prize was established in 1901 to honor Alfred Nobel, not Shannon.

2. **Content (Opinion/Emotion)**:
   - Sentence 1: "William was engaged while reading the paper on computer architecture."
   - Sentence 2: "William found the content of the paper on computer architecture to be extremely boring."
   Contradiction type: **Content (Opinion/Emotion)**
   Reason: The contradiction arises from different emotional responses to the same paper (engagement vs. boredom).

3. **Causal**:
   - Sentence 1: "David’s video, which was published on YouTube in 2024, was permanently removed that year due to violence."
   - Sentence 2: "Taylor came across this video published by David while browsing YouTube in 2025."
   Contradiction type: **Causal**
   Reason: The video was supposed to be removed in 2024, but it is still available in 2025.

4. **Negation**:
   - Sentence 1: "On April 20, 2025, the router experienced a brief routing disruption at 2:00 PM."
   - Sentence 2: "The router did not report any routing disruptions throughout April 20, 2025."
   Contradiction type: **Negation**
   Reason: The second sentence explicitly negates the occurrence of the routing disruption mentioned in the first sentence using "did not report any disruptions."

5. **Relation**:
   - Sentence 1: "The owner of the brand store, Howard, and his wife Sheila have a harmonious relationship after marriage."
   - Sentence 2: "Howard and his wife Emma have a beloved little daughter named Sheila."
   Contradiction type: **Relation**
   Reason: The first sentence implies Howard is married to Sheila, while the second sentence implies he is married to Emma, creating a contradiction in the relationships.

6. **Numeric**:
   - Sentence 1: "Howard’s brand store sold 29 pairs of women’s high heels on June 15, 2025."
   - Sentence 2: "Howard is concerned about the lack of any sales on June 15, 2025."
   Contradiction type: **Numeric**
   Reason: One sentence claims 29 pairs of high heels were sold, while the other suggests there were no sales at all.

7. **Temporal**:
   - Sentence 1: "Around 1860, Nobel and other researchers believed that nitroglycerin could not treat heart disease."
   - Sentence 2: "After Nobel’s death, further research revealed that nitroglycerin is effective for treating heart disease."
   Contradiction type: **Temporal**
   Reason: The belief about nitroglycerin changed over time, from it being ineffective in the 1860s to being effective after Nobel's death.

### Guidelines for classification:

- **Negation**: 
   - Any contradiction where one statement explicitly contradicts another through negation words or phrases such as "not", "never", "did not", "is not", "cannot", or "opposite".
   - Example: "John went to the party." vs "John did not go to the party."

- **Numeric**: 
   - Any contradiction involving numerical values, dates, or other quantitative measurements that are inconsistent across the statements.
   - Example: "The event had 500 attendees." vs "The event had over 1,000 attendees."

- **Temporal**: 
   - Any contradiction that arises because the knowledge or situation described has changed over time. This often involves a mismatch in dates or a temporal shift in facts.
   - Example: "In 2010, she graduated." vs "In 2015, she was still attending college."

### Output Format:
Return a JSON list:
[
  {
    "sentence_1": "...",
    "sentence_2": "...",
    "contradiction_type": "...",
    "reason": "..."
  },
  ...
]
If no contradictions are found, return: `No Contradiction Found.`
Be exhaustive — there may be multiple contradiction pairs.
"""


# -------------------------
# 提取跨文档矛盾
# -------------------------
def extract_cross_contradictions(text, llm):
    # Splitting the input text into two parts (assuming text1 is the first half and text2 is the second half)
    mid_index = len(text) // 2
    text1 = text[:mid_index].strip()
    text2 = text[mid_index:].strip()

    prompt = f"Compare the following two texts and find all contradictions:\n\nText 1:\n{text1}\n\nText 2:\n{text2}"
    resp = llm.generate(user_message=prompt, system_message=system_prompt)
    return resp.strip()


# -------------------------
# 保存函数
# -------------------------
def save_text(text, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
        f.write("\n")


# -------------------------
# 主函数
# -------------------------
def main():
    with open("text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 调用 LLM 获取跨文档矛盾
    result = extract_cross_contradictions(text, llm)

    # 如果没有矛盾，直接返回并打印提示
    if result == "No Contradiction Found.":
        print("No contradictions found between the texts.")
        return

    # 解析结果为 JSON
    try:
        contradictions = json.loads(result)

        # 过滤掉没有矛盾的句子对
        contradictions_with_conflict = [item for item in contradictions if item.get("contradiction_type") != "No Contradiction Found"]

        # 如果存在矛盾句子对
        if contradictions_with_conflict:
            # 保存包含所有矛盾的原始数据到 dense_text_cross.txt
            save_text(json.dumps(contradictions_with_conflict, ensure_ascii=False, indent=4), "dense_text_cross.txt")
            print("💾 Saved full contradiction data to dense_text_cross.txt")

            # 将有矛盾的句子对保存到单独的文件
            sentences1 = "\n".join([item["sentence_1"] for item in contradictions_with_conflict])
            sentences2 = "\n".join([item["sentence_2"] for item in contradictions_with_conflict])

            save_text(sentences1, "dense_text1.txt")
            save_text(sentences2, "dense_text2.txt")

            print("💾 Saved sentences with contradictions to dense_text1.txt and dense_text2.txt")
        else:
            print("No contradictory sentences to save.")

    except Exception as e:
        print("⚠️ Failed to parse JSON. Make sure LLM output is valid JSON.", e)


if __name__ == "__main__":
    main()


