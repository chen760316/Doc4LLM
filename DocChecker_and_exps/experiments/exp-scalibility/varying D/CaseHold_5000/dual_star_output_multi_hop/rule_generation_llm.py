"""
==================================================
           ✨ 规则生成任务处理完成 ✨
==================================================
📋 处理文件总数: 7
⏳ 总运行耗时: 35.99 秒 (约 0.60 分钟)
🚀 综合处理吞吐量: 11.67 文件/分钟
⏱️ 平均单文件周转时间 (并发状态): 5.14 秒
📅 结束时间: 2025-12-23 22:32:07
==================================================
"""
import json
import os
import re
from glob import glob
import asyncio
from functools import partial
from dotenv import load_dotenv
# 假设 knowledge_graph_maker.OpenAIClient 是同步实现的
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple, Set as TypeSet
import time  # 确保在文件顶部引入了 time 模块

# 设定目录
OUTPUT_DIR = "dual_star/llm_rules" # 统一使用这个目录

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

# -------------------- 异步包装器 --------------------

# 包装同步的 llm.generate 函数
async def _llm_generate_async(llm: OpenAIClient, user_message: str, system_message: str) -> str:
    """使用 asyncio.to_thread 包装同步的 LLM 调用。"""
    return await asyncio.to_thread(
        llm.generate,
        user_message=user_message,
        system_message=system_message
    )

# -------------------- 核心逻辑重构为异步 --------------------

def _extract_graph_data(graph_json_path: str) -> Tuple[List[str], str]:
    """同步函数：从 JSON 文件中提取数据和生成 prompt。"""
    with open(graph_json_path, "r", encoding="utf-8") as f:
        dual_star_data = json.load(f)

    if not isinstance(dual_star_data, dict) or "stars" not in dual_star_data:
        raise ValueError("输入 JSON 文件不是预期的双星结构（缺少 'stars' 键或不是字典）。")

    entity_descriptions = []
    for star in dual_star_data.get("stars", []):
        node_attrs_dict = star.get("node_attrs", {})
        multi_hop_node_attrs_dict = star.get("multi_hop_node_attrs", {})
        all_node_attrs = {**node_attrs_dict, **multi_hop_node_attrs_dict}

        for node_name, attrs in all_node_attrs.items():
            for attr, val in attrs.items():
                if attr not in ["name", "label", "text_span", "center"]:
                    entity_descriptions.append(f'{node_name}.{attr} = "{val}"')

        edges_attrs_dict = star.get("edges_attrs", {})
        multi_hop_edges_attrs_dict = star.get("multi_hop_edges_attrs", {})
        all_edge_attrs = {**edges_attrs_dict, **multi_hop_edges_attrs_dict}

        for relationship, props in all_edge_attrs.items():
            for prop_key, prop_val in props.items():
                if prop_key not in ["text_span", "doc_id"]:
                    entity_descriptions.append(f'"{relationship}".{prop_key} = "{prop_val}"')

    entity_descriptions = list(sorted(set(entity_descriptions)))

    # 构建用户提示的 graph_description 部分
    graph_description = chr(10).join(entity_descriptions)

    return entity_descriptions, graph_description


async def generate_predicates_with_llm_async(graph_json_path: str, llm: OpenAIClient, num_predicates: int = 100) -> List[str]:
    # 提取数据（同步操作）
    try:
        entity_descriptions, graph_description = _extract_graph_data(graph_json_path)
    except Exception as e:
        print(f"❌ 数据提取失败: {e}")
        return []

    # ... (系统消息和用户提示的构建保持不变) ...
    system_message = (
        "You are a knowledge graph inference engine. "
        "Given entities and relationships extracted from a document, "
        "generate expressive and diverse logical predicates. "
        "These predicates are used to construct RICO-style logical rules."
    )
    user_prompt = f"""Entities and relationships (where "Relationship" is the unique edge identifier):
{graph_description}
... (省略提示的其余部分以保持简洁) ...
Output only the {num_predicates} predicates, one per line.
"""

    response = await _llm_generate_async(llm, user_message=user_prompt, system_message=system_message)

    # *** 修正点：检查 response 是否为 None 或空字符串 ***
    if not response or not isinstance(response, str):
        print(f"❌ 警告: 文件 {os.path.basename(graph_json_path)} LLM 返回空响应，跳过谓词解析。")
        return []

    predicates = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return list(dict.fromkeys(predicates))


def build_rule_generation_prompt(rico_predicates: List[str], num_rules: int) -> str:
    # 示例规则为空列表，因此我们简化提示
    return f"""You are an expert in logical reasoning and contradiction detection.

Your task is to generate RICO-style rules based on the given list of predicates.

Each rule should follow the format:
Q[x0, y0](P1 ∧ P2 ∧ ... → P_out)

### Rule Constraints:

1. Each rule must include between **2 and 6 predicate conditions** in the premise.
2. You must generate a **diverse set of rules** with varying premise lengths — specifically, make sure to include rules whose premises have 2, 3, 4, 5, and 6 predicate conditions.
3. The number of rules for each premise length should be **as evenly distributed as possible** — for example, if you are generating {num_rules} rules, try to generate about {num_rules // 5} rules for each premise length (length 2 to 6).
4. Each rule must have **exactly one predicate in the conclusion**.
5. The **conclusion predicate must not be identical to any of the predicates in the premise**.

--- Predicates ---
{chr(10).join(rico_predicates)}

--- Now write exactly {num_rules} new rules below ---
Rules:"""


async def generate_rico_predicates_and_rules_concurrent(
    graph_json_path: str,
    output_path: str,
    llm: OpenAIClient,
    num_predicates: int = 100,
    num_rules: int = 30 # 统一使用30，因为原始提示中要求30条规则
):
    print(f"[START] {os.path.basename(graph_json_path)}")

    # === Step 1: Generate predicates from graph (Async LLM Call) ===
    predicates = await generate_predicates_with_llm_async(graph_json_path, llm, num_predicates=num_predicates)

    if not predicates:
        print(f"[FAIL] {os.path.basename(graph_json_path)}: 谓词生成失败或为空。")
        return

    # === Step 2: Generate rules based on predicates (Async LLM Call) ===
    rules_prompt = build_rule_generation_prompt(predicates, num_rules=num_rules)
    system_message = (
        "You are a natural language inference engine that generates RICO-style logical rules "
        "from given predicates. Each rule should follow the format Q[x0, y0](P1 ∧ P2 ∧ ... → P_out) "
        "and capture a plausible implication or contradiction."
    )

    response = await _llm_generate_async(llm, user_message=rules_prompt, system_message=system_message)

    rules = [line.strip() for line in response.strip().splitlines() if line.strip()]

    # === Step 3: Save everything (同步操作) ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Predicates:\n")
        for p in predicates:
            f.write(p + "\n")
        f.write("\nRules:\n")
        for r in rules:
            f.write(r + "\n")

    print(f"✅ [SUCCESS] {os.path.basename(graph_json_path)}: Saved {len(predicates)} predicates and {len(rules)} rules.")


# -------------------- III. 异步主函数 --------------------

async def main_concurrent():
    # 设置输入和输出目录
    INPUT_DIR = "dual_star_json_with_attrs"

    # 查找所有JSON文件
    graph_files = glob(os.path.join(INPUT_DIR, "*.json"))

    if not graph_files:
        print(f"⚠️ 警告: 在目录 {INPUT_DIR} 中未找到任何 JSON 文件。请检查路径。")
        return

    # === ⏱️ 记录开始时间 ===
    start_time = time.perf_counter()
    start_wall_time = time.strftime('%H:%M:%S', time.localtime())

    print(f"--- 🚀 任务启动时间: {start_wall_time} ---")
    print(f"--- ⚙️ 发现 {len(graph_files)} 个文件，正在使用并发模式处理... ---")

    tasks = []
    for graph_json_path in graph_files:
        base_name = os.path.basename(graph_json_path).replace('.json', '')
        output_file_path = os.path.join(OUTPUT_DIR, f"{base_name}_rules_llm.txt")

        task = generate_rico_predicates_and_rules_concurrent(
            graph_json_path=graph_json_path,
            output_path=output_file_path,
            llm=llm,
            num_predicates=100,
            num_rules=30
        )
        tasks.append(task)

    # 并行执行所有任务
    await asyncio.gather(*tasks)

    # === ⏱️ 记录结束时间 ===
    end_time = time.perf_counter()
    total_duration = end_time - start_time

    # === 📊 生成统计报告 ===
    print("\n" + "=" * 50)
    print("           ✨ 规则生成任务处理完成 ✨")
    print("=" * 50)
    print(f"📋 处理文件总数: {len(graph_files)}")
    print(f"⏳ 总运行耗时: {total_duration:.2f} 秒 (约 {total_duration / 60:.2f} 分钟)")

    if len(graph_files) > 0:
        # 并发环境下的平均指标
        avg_speed = total_duration / len(graph_files)
        throughput = (len(graph_files) / total_duration) * 60
        print(f"🚀 综合处理吞吐量: {throughput:.2f} 文件/分钟")
        print(f"⏱️ 平均单文件周转时间 (并发状态): {avg_speed:.2f} 秒")

    print(f"📅 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    print("=" * 50)


if __name__ == "__main__":
    # 使用 asyncio.run 启动异步主函数
    asyncio.run(main_concurrent())