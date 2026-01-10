"""
首先从原始文本中提取完整的三元组，然后在所有三元组中筛选与用户指定实体相关的三元组
向用户筛选的三元组中头实体和尾实体对应的上下文中注入冲突
"""
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# 加载 API 密钥
load_dotenv()
# 初始化 OpenAI 客户端（新版接口）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def parse_triples_and_context(filepath):
    """
    解析文本文件，返回三元组列表和对应上下文字典
    假设文件格式如下（示例）：
    Triple 1: (Scrooge) -[kept an eye on]-> (clerk)
    Head entity sentences for 'Scrooge':
    ...
    Tail entity sentences for 'clerk':
    ...
    ============================================================
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    triples = []
    # 正则提取三元组
    triple_pattern = re.compile(r"Triple \d+: \((.*?)\) -\[(.*?)\]-> \((.*?)\)", re.DOTALL)
    # 正则提取上下文
    head_context_pattern = re.compile(r"Head entity sentences for '(.+?)':\n(.*?)\n\nTail entity sentences for", re.DOTALL)
    tail_context_pattern = re.compile(r"Tail entity sentences for '(.+?)':\n(.*?)(?:\n\n=+|\Z)", re.DOTALL)

    triple_matches = triple_pattern.findall(content)
    head_contexts = head_context_pattern.findall(content)
    tail_contexts = tail_context_pattern.findall(content)

    # 建立实体上下文映射
    head_context_map = {ent: ctx.strip() for ent, ctx in head_contexts}
    tail_context_map = {ent: ctx.strip() for ent, ctx in tail_contexts}

    for h, r, t in triple_matches:
        head_ctx = head_context_map.get(h, "")
        tail_ctx = tail_context_map.get(t, "")
        triples.append({
            "head": h,
            "relation": r,
            "tail": t,
            "head_context": head_ctx,
            "tail_context": tail_ctx,
        })
    return triples

def generate_conflict_text(triple):
    prompt = f"""
You are a text editing expert.

Given the known fact:
Head entity: {triple['head']}
Relation: {triple['relation']}
Tail entity: {triple['tail']}

Context related to the head entity:
{triple['head_context']}

Context related to the tail entity:
{triple['tail_context']}

Based on the above contexts and fact, please generate a new paragraph of text that **directly contradicts the given fact**.
In other words, the new text should express something opposite or inconsistent with the original relation, while keeping the context natural and fluent.

Please output only the conflicting English paragraph. Do NOT include any explanations, instructions, or Chinese text.
"""
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=100,
    )
    conflict_text = response.choices[0].message.content.strip()
    return conflict_text



def main():
    filepath = "entity_full_contexts.txt"  # 你的文件名
    output_path = "entity_conflict_contexts.txt"
    triples = parse_triples_and_context(filepath)
    print(f"共解析到 {len(triples)} 条三元组及上下文")

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, triple in enumerate(triples, 1):
            print(f"\n=== 处理三元组 {i} / {len(triples)} ===")
            print(f"三元组: ({triple['head']}) -[{triple['relation']}]-> ({triple['tail']})")

            conflict_text = generate_conflict_text(triple)

            # 写入文件
            fout.write(f"=== Triple {i} ===\n")
            fout.write(f"Triple: ({triple['head']}) -[{triple['relation']}]-> ({triple['tail']})\n")
            fout.write("Conflict Text:\n")
            fout.write(conflict_text + "\n")
            fout.write("\n" + "="*50 + "\n")

            print(f"已保存冲突文本到文件")

if __name__ == "__main__":
    main()
