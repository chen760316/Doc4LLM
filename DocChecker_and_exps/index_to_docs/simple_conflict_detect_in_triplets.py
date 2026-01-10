"""
首先提取文本中的三元组，然后借助大模型围绕三元组检测冲突
冲突检测的简单实现，直接借助ML模型或LLM从文档中提取实体冲突属性，以及判断文档中实体的不一致关系。
本文处理statements, sections, and even documents之间的冲突和差异，以及单词和短语的固有模糊性
Conflicting Statements	是否存在关于同一实体的矛盾描述
Uncertain Language	是否使用了模糊性、不确定性语言（如 perhaps, might, looked like）
Mentioned Entities Involved	出现在该 chunk 中、与前两类信息有关的实体
"""

import os
from dotenv import load_dotenv
from src.kg_gen import KGGen
from openai import OpenAI
import tiktoken

# 参数配置
MAX_TOKENS = 512
STRIDE = 64

# 初始化 OpenAI 客户端
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# 初始化 KGGen 实体关系抽取模型
kg = KGGen(
    api_key=os.getenv("OPENAI_API_KEY")
)


def split_text_by_token_limit(text, max_tokens=512, stride=64, model_name="gpt-5.2"):
    """分割文本（带滑动窗口）"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - stride
    return chunks


def ask_conflict_about_triples(text_chunk, triples):
    """向 GPT-4o 查询：这些三元组是否存在冲突或模糊性"""
    triple_text = "\n".join([f"({h}) -[{r}]-> ({t})" for h, r, t in triples])
    prompt = f"""
You are a knowledge graph and text reasoning expert.

Given the following text chunk and the extracted triples, identify:
1. If any conflicting statements or logical contradictions exist among the triples;
2. If any triple or part of it is vague, speculative, or semantically ambiguous.

Return in structured format:
- Conflicting Triples: [...]
- Uncertain/Vague Triples: [...]
- Entities Involved in Conflict: [...]

Text:
\"\"\"
{text_chunk}
\"\"\"

Triples:
{triple_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error querying GPT-4o: {e}]"


if __name__ == "__main__":
    # 加载输入文本
    with open("book_chunk_conflict.txt", "r", encoding="utf-8") as file:
        full_text = file.read()

    # 文本分块
    chunks = split_text_by_token_limit(full_text, max_tokens=MAX_TOKENS, stride=STRIDE)

    print(f"\nProcessing {len(chunks)} chunks for KG + Conflict Detection...\n")
    final_results = []
    all_relations = []

    for i, chunk in enumerate(chunks):
        print(f"🔍 Extracting triples from chunk {i + 1}/{len(chunks)}...")

        try:
            graph = kg.generate(input_data=chunk, model="gpt-5.2")
            if graph and hasattr(graph, 'relations') and graph.relations:
                all_relations.extend(graph.relations)
                triples = graph.relations
                print(f"✅ Extracted {len(triples)} triples. Now analyzing conflicts...")
                print(triples)
                analysis = ask_conflict_about_triples(full_text, triples)
                print(f"\n📘 Conflict Analysis for Chunk {i + 1}:\n{analysis}\n{'-'*80}\n")
                final_results.append(f"Chunk {i+1}:\n{analysis}")
            else:
                print("⚠️ No triples extracted.")
        except Exception as e:
            print(f"❌ Error in chunk {i + 1}: {e}")

    # 保存结果
    with open("conflict_from_kg_report.txt", "w", encoding="utf-8") as out_file:
        out_file.write("\n\n".join(final_results))

    print("✅ All chunks processed. Conflict report saved to 'conflict_from_kg_report.txt'")

    # Print all extracted triples
    print("\nFinal Extracted Triples:")
    for i, (head, relation, tail) in enumerate(all_relations, 1):
        print(f"{i}. ({head}) -[{relation}]-> ({tail})")
