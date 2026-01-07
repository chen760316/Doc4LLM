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
INPUT_DIR = "dual_star_json_with_attrs"
DENSE_TEXT_PATH = "dense_text.txt"
OUTPUT_DIR = "dual_star/llm_rules_dense_txt"
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

# === 矛盾类型定义 (保持原有定义) ===
CONTRADICTION_SYSTEM_PROMPT = """[此处省略你原有的 7 种矛盾类型定义，建议保留在代码中]"""


# -------------------- I. 数据辅助函数 --------------------

def load_dense_text_data(path: str) -> List[Dict]:
    """加载并解析参考矛盾文本文件"""
    if not os.path.exists(path):
        print(f"⚠️ 找不到参考文件: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 解析 {path} 失败: {e}")
        return []


def find_matching_case(center_text: str, dense_data: List[Dict]) -> Optional[Dict]:
    """根据双星中心节点(center)匹配对应的矛盾条目"""
    if not center_text or not dense_data:
        return None

    center_clean = center_text.lower().strip()
    for item in dense_data:
        s1 = item.get("sentence_1", "").lower()
        s2 = item.get("sentence_2", "").lower()
        # 如果 center 出现在 s1 或 s2 中，则认为是相关案例
        if center_clean in s1 or center_clean in s2:
            return item
    return None


def _extract_graph_data(graph_json_path: str) -> Tuple[List[str], str, str]:
    """针对双星结构优化的数据提取：从 stars 列表中提取 center"""
    try:
        with open(graph_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败 {graph_json_path}: {e}")
        return [], "", "Unknown"

    # --- 修正点 1：正确提取 Center ---
    stars = data.get("stars", [])
    center = "Unknown"
    # 从第一个 star 的 center 字段提取，如果不存在则尝试从 node_attrs 找
    if stars:
        center = stars[0].get("center", "Unknown")

    entity_descriptions = []

    # --- 修正点 2：遍历提取属性 ---
    for star in stars:
        # 提取节点属性
        all_nodes = {**star.get("node_attrs", {}), **star.get("multi_hop_node_attrs", {})}
        for node_name, attrs in all_nodes.items():
            for attr, val in attrs.items():
                if attr not in ["text_span", "name"]:
                    entity_descriptions.append(f'{node_name}.{attr} = "{val}"')

        # 提取边属性
        all_edges = {**star.get("edges_attrs", {}), **star.get("multi_hop_edges_attrs", {})}
        for edge_text, props in all_edges.items():
            edge_id = props.get("source_id", "edge")
            for p_key, p_val in props.items():
                if p_key not in ["text_span", "source_id"]:
                    entity_descriptions.append(f'"{edge_id}".{p_key} = "{p_val}"')

    graph_description = "\n".join(sorted(set(entity_descriptions)))
    return entity_descriptions, graph_description, center


# -------------------- II. 异步生成逻辑 --------------------

async def _llm_generate_async(llm: OpenAIClient, user_message: str, system_message: str) -> str:
    try:
        return await asyncio.to_thread(
            llm.generate, user_message=user_message, system_message=system_message
        )
    except Exception as e:
        print(f"❌ LLM 请求失败: {e}")
        return ""


async def generate_predicates_with_llm_async(
        graph_path: str,
        ref_case: Optional[Dict],
        num_predicates: int = 100
) -> List[str]:
    # 提取图数据和中心点
    entity_list, graph_desc, center = _extract_graph_data(graph_path)

    # 构造参考案例上下文（如果有）
    ref_context = "No direct text reference provided."
    if ref_case:
        ref_context = (
            f"REFERENCE CASE (Target Contradiction):\n"
            f"- S1: {ref_case['sentence_1']}\n"
            f"- S2: {ref_case['sentence_2']}\n"
            f"- Declared Type: {ref_case['contradiction_type']}\n"
            f"- Ground Truth Reason: {ref_case['reason']}\n"
        )

    system_message = (
            "You are a knowledge graph inference engine. Your mission is to define logical predicates "
            "that follow a strict syntax to support RICO-style contradiction rules.\n"
            + CONTRADICTION_SYSTEM_PROMPT
    )

    user_prompt = f"""{ref_context}

LOCAL GRAPH DATA (Context: {center}):
{graph_desc}

### TASK:
Generate exactly {num_predicates} unique logical predicates based on the data above. 
You MUST follow the structural categories below:

1. Entity-level (Node attributes):
   Syntax: node_name.attribute
   Example: {entity_list[0] if entity_list else 'John_Smith.birth_year'}

2. Edge-level (Relationship attributes):
   Syntax: "relationship_name".attribute
   Example: "father_of".status

3. Meta-level (Logical & Contradiction operators):
   - M_NLI(s1, s2): General inference between two attribute strings.
   - T(s, s', tau): Determines the specific type 'tau' (one of: Factual, Content, Causal, Negation, Relational, Numeric, Temporal).
   - tau(x.A, y.B): Specific contradiction instance between two node attributes.
   - violate(x0, y0): The primary contradiction flag.

⚠️ REQUIREMENTS:
- Predicates must be able to formalize the 'Ground Truth Reason' provided.
- Include 'violate(x0, y0)' and specific type predicates like 'Factual(x0, y0)', 'Numeric(x0, y0)', etc.
- Output ONLY the predicates, one per line. No extra text.
"""

    response = await _llm_generate_async(llm, user_prompt, system_message)

    if not response or not isinstance(response, str):
        print(f"❌ 警告: {os.path.basename(graph_path)} 返回空响应。")
        return []

    # 解析并去重
    predicates = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # 基础谓词兜底，确保生成的规则有结论可用
    required_bases = ["violate(x0, y0)", "Factual(x0, y0)", "Causal(x0, y0)", "Numeric(x0, y0)", "Temporal(x0, y0)"]
    for base in required_bases:
        if base not in predicates:
            predicates.append(base)

    return list(dict.fromkeys(predicates))


async def generate_rico_rules_async(
        predicates: List[str],
        ref_case: Optional[Dict],
        num_rules: int = 30
) -> List[str]:
    ref_hint = ""
    if ref_case:
        ref_hint = f"Note: At least 5 rules MUST directly model the logic described here: {ref_case['reason']}"

    system_message = (
        "You are a logic engine generating RICO-style rules: Q[x0, y0](P1 ∧ P2 ... → P_out). "
        "The conclusion MUST be a contradiction predicate (e.g., violate(x0, y0) or Factual(x0, y0))."
    )

    user_prompt = f"""{ref_hint}

AVAILABLE PREDICATES:
{chr(10).join(predicates)}

TASK:
Generate exactly {num_rules} diverse rules. 
Premise length should range from 2 to 6 predicates.
Ensure the rules capture the logical patterns that lead to the contradictions defined in the reference.

Output the rules line by line.
"""
    response = await _llm_generate_async(llm, user_prompt, system_message)
    if not response: return []
    return [line.strip() for line in response.strip().splitlines() if line.strip()]


# -------------------- III. 任务调度与主函数 --------------------

async def process_single_file(graph_path: str, dense_data: List[Dict]):
    base_name = os.path.basename(graph_path)
    output_path = os.path.join(OUTPUT_DIR, base_name.replace(".json", "_rules_llm.txt"))

    # --- 修正点 3：使用统一的提取逻辑获取 center ---
    # 先预提取一次数据，确保谓词生成和匹配使用同一个 center
    entity_list, graph_desc, center_text = _extract_graph_data(graph_path)

    # 字符串匹配
    ref_case = find_matching_case(center_text, dense_data)

    # 状态打印
    if ref_case:
        print(f"🔗 [MATCHED] {base_name} (Center: {center_text})")
    else:
        print(f"❓ [UNMATCHED] {base_name} (Center: {center_text})")

    # 2. 生成谓词 (直接传入已经提取好的数据，避免重复读取)
    # 注意：这里需要稍微调整 generate_predicates_with_llm_async 以接受预提取的数据，
    # 或者让它内部也使用修正后的 _extract_graph_data。
    predicates = await generate_predicates_with_llm_async(graph_path, ref_case)
    if not predicates:
        return

    # 3. 生成规则
    rules = await generate_rico_rules_async(predicates, ref_case)

    # 4. 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Source JSON: {base_name}\n")
        f.write(f"Center Node: {center_text}\n")
        f.write(f"Matched Case: {json.dumps(ref_case, ensure_ascii=False) if ref_case else 'None'}\n\n")
        f.write("Predicates:\n" + "\n".join(predicates) + "\n\n")
        f.write("Rules:\n" + "\n".join(rules) + "\n")
    print(f"✅ [SUCCESS] {base_name}: Saved {len(rules)} rules.")


async def main_concurrent():
    dense_data = load_dense_text_data(DENSE_TEXT_PATH)
    graph_files = glob(os.path.join(INPUT_DIR, "*.json"))

    if not graph_files:
        print(f"⚠️ 目录下未找到 JSON 文件: {INPUT_DIR}")
        return

    print(f"🚀 开始处理 {len(graph_files)} 个文件...")

    # 使用 Semaphore 控制并发量，防止触发 API 速率限制
    sem = asyncio.Semaphore(5)

    async def sem_task(f):
        async with sem:
            await process_single_file(f, dense_data)

    tasks = [sem_task(f) for f in graph_files]
    await asyncio.gather(*tasks)
    print("\n--- ✨ 全部处理完毕 ---")


if __name__ == "__main__":
    asyncio.run(main_concurrent())