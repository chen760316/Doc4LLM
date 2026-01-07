import json
import os
import re
from glob import glob
import asyncio
from functools import partial
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient

# === 配置区域 ===
DENSE_TEXT_PATH = "dense_text.txt"
OUTPUT_DIR = "dual_star/generate_rules_dense_txt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# === 核心系统提示词 (严格限定 tau 类型) ===
CONTRADICTION_SYSTEM_PROMPT = """
The 7 types of contradiction predicates (tau):
- Factual(x, y), Content(x, y), Causal(x, y), Negation(x, y), Relational(x, y), Numeric(x, y), Temporal(x, y).
"""


async def generate_logic_content(case: Dict, index: int):
    s1, s2 = case['sentence_1'], case['sentence_2']
    reason = case['reason']
    tau_type = case.get('contradiction_type', 'Factual')

    # --- 第一步：生成高质量谓词 ---
    predicates_prompt = f"""
S1: {s1}
S2: {s2}
Reason: {reason}

TASK: Generate 30 unique logical predicates based on the data.
1. Entity nodes: node_name.attribute (e.g., Publication_1928.year)
2. Edge nodes: "relationship_name".attribute
3. Meta: M_NLI(s1, s2), {tau_type}(x0, y0), violate(x0, y0)
Output ONLY predicates, one per line.
"""
    predicates_raw = await asyncio.to_thread(
        llm.generate,
        user_message=predicates_prompt,
        system_message="You are a logic parser. Output valid predicates only."
    )
    predicates = [p.strip() for p in (predicates_raw or "").strip().splitlines() if p.strip()]
    if not predicates:
        predicates = ["violate(x0, y0)", f"{tau_type}(x0, y0)"]

    # --- 第二步：生成强逻辑规则 ---
    # 提取 S1 中的第一个有效词作为推荐锚点，减少评估时的匹配失败
    potential_anchor = (re.findall(r'\b\w{3,}\b', s1) + ["Context"])[0]

    rules_prompt = f"""
### CONTEXT:
S1: {s1}
S2: {s2}
Target Reason: {reason}
Target Conclusion: {tau_type}(x0, y0)

### TASK:
Generate 30 RICO rules: Q[Anchor, x0, y0](Premise → Conclusion).

⚠️ STRICT REQUIREMENTS FOR 85%+ COVERAGE:
1. 'Anchor' MUST be an EXACT word from S1 or S2 (e.g., '{potential_anchor}'). NO UNDERSCORES.
2. Premise MUST use specific predicates like x0.attr == "value".
3. The rule must logically prove why S1 and S2 contradict based on the Reason.
4. Conclusion MUST be {tau_type}(x0, y0).

AVAILABLE PREDICATES:
{chr(10).join(predicates)}
"""
    rules_raw = await asyncio.to_thread(
        llm.generate,
        user_message=rules_prompt,
        system_message="You are a logic engine that outputs RICO rules."
    )
    rules = [r.strip() for r in (rules_raw or "").strip().splitlines() if "Q[" in r]

    return predicates, rules


async def process_case(case: Dict, idx: int):
    case_id = f"case_{idx:03d}"
    print(f"🔄 Processing {case_id}...")

    predicates, rules = await generate_logic_content(case, idx)

    output_path = os.path.join(OUTPUT_DIR, f"{case_id}_rules.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Source: dense_text.txt (Index {idx})\n")
        f.write(f"S1: {case['sentence_1']}\n")
        f.write(f"S2: {case['sentence_2']}\n\n")
        f.write("Predicates:\n" + "\n".join(predicates) + "\n\n")
        f.write("Rules:\n" + "\n".join(rules) + "\n")
    print(f"✅ {case_id} completed.")


async def main():
    if not os.path.exists(DENSE_TEXT_PATH): return
    with open(DENSE_TEXT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    sem = asyncio.Semaphore(5)

    async def sem_task(case, i):
        async with sem: await process_case(case, i)

    await asyncio.gather(*[sem_task(case, i) for i, case in enumerate(data)])


if __name__ == "__main__":
    asyncio.run(main())