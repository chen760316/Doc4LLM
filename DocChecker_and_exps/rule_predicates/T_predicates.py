"""
谓词T(s, s′,\tau)的实现，首先判断实体或边转换成的两个句子之间的矛盾类型
随后根据矛盾的解释可以定位 \tau(x.A, 𝑦.B)和\tau(ex.A, e𝑦.B)，即解释中明显出现矛盾的节点或边的属性
判断两个星型模式是否存在以下7种矛盾类型之一：

借助llm谓词区分以下七种矛盾类型：
1、Factual ：需要外部世界的知识来证实矛盾（检测的时候需要外部知识库支持）
2、Opinion/Emotion/Content/Misinformation/Semantic/Lexical  Conﬂict：
更改事件或实体的一个或多个属性造成的矛盾，其中Opinion/Emotion是content在情态属性上的特例，Misinformation等价于Content，
Semantic/Lexical是指多义词在理解上引发的歧义现象，是Content针对语言多义混淆情景下的特例
3、Causal：结果与原因不相符引发矛盾（一定是有明确的因果关系）
4、Negation/Antonym：否定原句子（一定包含明确的否定关系，比如正反义词，do not , is not等否定词汇的使用）
5、Relational/Structure： 实体之间两个互斥关系的描述（一定是实体之间的关系冲突），包含关系的颠倒，缺失，关系两侧实体不匹配等类型
6、Numeric：数字不匹配或数字超出范围（一定存在数字属性的冲突）
7、Temporal：知识随时间推移发生变化或更新时所引发的矛盾（一定存在时间顺序上的属性或关系不一致）

新版
useful
"""
import json
from knowledge_graph_maker import OpenAIClient
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4.1-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# 定义系统提示词，包含矛盾类型说明和示例
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
   
8. **Negation**: Contradictions resulting from negating the original sentence (must involve explicit negation, such as antonyms or negative words like "do not," "is not").
   Example:
   - Sentence 1: "Zully donated her kidney."
   - Sentence 2: "Zully never donated her kidney."
   Contradiction type: Negation, because the second sentence directly negates the claim made in the first sentence by using "never."

9. **Numeric**: Contradictions due to mismatched numbers or numbers falling outside an acceptable range.
   Example:
   - Sentence 1: "All the donors are between 20 to 45 years old."
   - Sentence 2: "Lisa, who donates her kidney, she is 70 years old."
   Contradiction type: **Numeric**, because one sentence sets a range of ages (20 to 45 years old) for the donors, while the other mentions a donor (Lisa) who is 70 years old, which falls outside the specified range.

10. **Negation**: Contradictions resulting from negating the original sentence (must involve explicit negation, such as antonyms or negative words like "do not," "is not").
   Example:
   - Sentence 1: "Zully Broussard donated her kidney to a stranger."
   - Sentence 2: "Zully Broussard donated her kidney to her close friend."
   Contradiction type: **Negation**, because the second sentence directly contradicts the first by changing the recipient of the kidney donation from a stranger to a close friend, effectively negating the original information.

11. **Content**: Contradictions caused by altering one or more attributes of an event or entity. This includes:
   - **Opinion/Emotion**: A contradiction involving differing emotional reactions or opinions.
   Example:
   - Sentence 1: "The doctor spoke highly of the project and called it 'a breakthrough.'"
   - Sentence 2: "The doctor disliked the project, saying it had no impact at all."
   Contradiction type: Content (Opinion/Emotion), because the contradiction arises from the differing emotional reactions of the doctor toward the project. The first sentence expresses strong approval ("a breakthrough"), while the second expresses strong disapproval (no impact at all).

12. **Content**: Contradictions caused by altering one or more attributes of an event or entity. This includes:
   - **Opinion/Emotion**: A contradiction involving differing emotional reactions or opinions.
   Example:
   - Sentence 1: "The rescue team searched for the boy worriedly."
   - Sentence 2: "The rescue team searched for the boy happily."
   Contradiction type: Content (Opinion/Emotion), because the emotional response to the same event (searching for the boy) is contradictory (worried vs. happy).

13. **Relation**: Contradictions arising from descriptions of two mutually exclusive relationships between entities (must involve conflicts in relationships between entities), including reversed, missing, or mismatched relationships.
   Example:
   - Sentence 1: "Jane and Tom are a married couple."
   - Sentence 2: "Jane is Tom’s sister."
   Contradiction type: Relation, because the first sentence suggests Jane and Tom are married, while the second sentence contradicts that by stating Jane is Tom’s sister, which is a mutually exclusive relationship.
   
14. **Factual**: External knowledge about the location of a road.
   Example:
   - Sentence 1: "The road T51 was located in New York."
   - Sentence 2: "The road T51 was located in California."
   Contradiction type: **Factual**, because this contradiction requires factual verification about the actual location of road T51, which can be confirmed through an external map or database.

15. **Causal**: Contradictions arising from a mismatch between cause and effect (must involve a clear causal relationship).
   Example:
   - Sentence 1: "I slam the door."
   - Sentence 2: "After I do that, the door opens."
   Contradiction type: Causal, because slamming the door should logically cause the door to close, not open, creating a mismatch in the cause and effect relationship.

If the sentences are contradictory, output ONLY the contradiction type: one of ["Factual", "Content", "Causal", "Negation", "Relation", "Numeric", "Temporal"]. Include the reasoning for the contradiction after the type. If the sentences are **NOT contradictory**, output the phrase "No Contradiction" along with the reason why they do not contradict.
"""


def classify_contradiction_type(pair, llm):
    prompt = f"""
Sentence 1: "{pair['sentence_1']}"
Sentence 2: "{pair['sentence_2']}"

What type of contradiction do these sentences represent? Please explain why.
"""
    response = llm.generate(
        user_message=prompt,
        system_message=system_prompt
    )
    return response.strip()

def detect_contradiction(pair, llm):
    prompt = f"""
Sentence 1: "{pair['sentence_1']}"
Sentence 2: "{pair['sentence_2']}"

Do these sentences contradict each other? 
Answer only "Yes" or "No".
"""
    system_prompt = (
        "You are a language reasoning assistant. "
        "Only answer 'Yes' if the sentences are contradictory, otherwise 'No'."
    )
    response = llm.generate(
        user_message=prompt,
        system_message=system_prompt
    )
    return response.strip().lower() == 'yes'

def main():
    """原始的用于学习不同矛盾类型的句子对，现注释"""
    # pairs = [
    #     {"sentence_1": "Zully donated her kidney.", "sentence_2": "Zully never donated her kidney."},  # Negetion
    #     {"sentence_1": "All the donors are between 20 to 45 years old.", "sentence_2": "Lisa, who donates her kidney, she is 70 years old."},  # Numeric
    #     {"sentence_1": "Zully Broussard donated her kidney to a stranger.", "sentence_2": "Zully Broussard donated her kidney to her close friend."},  # Content
    #     {"sentence_1": "The doctor spoke highly of the project and called it “a breakthrough”.", "sentence_2": "The doctor disliked the project, saying it had no impact at all."},  # Content（Opinion)
    #     {"sentence_1": "The rescue team searched for the boy worriedly.", "sentence_2": "The rescue team searched for the boy happily."},  # Content（Emotion)
    #     {"sentence_1": "Jane and Tom are a married couple.", "sentence_2": "Jane is Tom’s sister."},  # Relation
    #     {"sentence_1": "The road T51 was located in New York.", "sentence_2": "The road T51 was located in California."},  # Factual
    #     {"sentence_1": "I slam the door.", "sentence_2": "After I do that, the door opens."},  # Causal
    #     {"sentence_1": "The Nobel Prize was indeed established in 1901 to commemorate Alfred Nobel.", "sentence_2": "Hinton received a world-renowned scientific award in 2024, which was established in 1901 to honor Shannon's contributions to the field of information."},  # Factual
    #     {"sentence_1": "William was engaged while reading the paper on computer architecture.", "sentence_2": "William found the content of the paper on computer architecture to be extremely boring."},  # Content
    #     {"sentence_1": "David’s video, which was published on YouTube in 2024, was permanently removed that year due to violence.", "sentence_2": "Taylor came across this video published by David while browsing YouTube in 2025."},  # Casual
    #     {"sentence_1": "On April 20, 2025, the router experienced a brief routing disruption at 2:00 PM.", "sentence_2": "The router did not report any routing disruptions throughout April 20, 2025."},  # Negation
    #     {"sentence_1": "The owner of the brand store, Howard, and his wife Sheila have a harmonious relationship after marriage.", "sentence_2": "Howard and his wife Emma have a beloved little daughter named Sheila."},  # Relational
    #     {"sentence_1": "Howard’s brand store sold 29 pairs of women’s high heels on June 15, 2025.", "sentence_2": "Howard is concerned about the lack of any sales on June 15, 2025."},  # Numeric
    #     {"sentence_1": "Around 1860, Nobel and other researches believed that nitroglycerin could not treat heart disease.", "sentence_2": "After Nobel’s death, further research revealed that nitroglycerin is effective for treating heart disease."}  # Temporal
    # ]

    pairs = [
    {"sentence_1": "Einstein developed the theory of relativity in the early 20th century.", "sentence_2": "Einstein's theory of relativity was first published in 1920."}, # Factual
    {"sentence_1": "She loves spending time at the beach, especially during the summer.", "sentence_2": "She hates going to the beach, finding it too crowded in the summer."},  # Content (Opinion/Emotion)
    {"sentence_1": "The earthquake shook the building violently.", "sentence_2": "After the earthquake, the building remained perfectly intact."}, # Causal
    {"sentence_1": "The flight to Paris was delayed by 2 hours.", "sentence_2": "The flight to Paris was not delayed at all."},  # Negation
    {"sentence_1": "John and Sarah have been best friends for over a decade.", "sentence_2": "John and Sarah are married and have two children together."},  # Relation
    {"sentence_1": "The stadium holds 50,000 spectators.", "sentence_2": "Only 12,000 people attended the game last night."},  # Numeric
    {"sentence_1": "The movie was released in 1995 and became an instant classic.", "sentence_2": "The movie was first released in 2000 and became an instant classic."},  # Temporal
    {"sentence_1": "He was overjoyed when he received the promotion.", "sentence_2": "He felt devastated after being promoted, worried about the new responsibilities."},  # Content (Emotion)
    {"sentence_1": "The Eiffel Tower is located in Paris, France.", "sentence_2": "The Eiffel Tower is located in Berlin, Germany."},  # Factual
    {"sentence_1": "I forgot to water the plants, so they all died.", "sentence_2": "Despite forgetting to water them, the plants grew beautifully."},  # Causal
    {"sentence_1": "David and Laura are childhood friends.", "sentence_2": "David and Laura are cousins, and they grew up together."},  # Relation
    {"sentence_1": "The book was written by Mary Shelley in 1818.", "sentence_2": "The book was written by George Orwell in 1949."},  # Relation
    {"sentence_1": "Zully donated her kidney.", "sentence_2": "Zully never donated her kidney."},  # Negation
    {"sentence_1": "Zully Broussard donated her kidney to a stranger.", "sentence_2": "Zully Broussard donated her kidney to her close friend."},  # Content
    {"sentence_1": "The doctor spoke highly of the project and called it 'a breakthrough.'", "sentence_2": "The doctor disliked the project, saying it had no impact at all."},  # Content (Opinion)
    {"sentence_1": "The rescue team searched for the boy worriedly.", "sentence_2": "The rescue team searched for the boy happily."},  # Content (Emotion)
    {"sentence_1": "Jane and Tom are a married couple.", "sentence_2": "Jane is Tom’s sister."},  # Relation
]

    for pair in pairs:
        if detect_contradiction(pair, llm):
            contradiction_type = classify_contradiction_type(pair, llm)
        else:
            contradiction_type = "No Contradiction"
        print(f"Sentences:\n  1: {pair['sentence_1']}\n  2: {pair['sentence_2']}\nContradiction type: {contradiction_type}\n")


if __name__ == "__main__":
    main()

