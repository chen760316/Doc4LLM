import os
import json
import time
import asyncio
from glob import glob
from collections import Counter
from itertools import combinations
from functools import partial
from typing import List, Dict, Any, Tuple, Set

# -------------------- I. 配置与环境 --------------------
INPUT_DIR = "dual_star_json_with_attrs"
OUTPUT_DIR = "dual_star/mined_support_rules"
MIN_SUPPORT = 2       # 最小支持度：规则至少在2个星型结构中出现
MIN_CONFIDENCE = 0.5  # 最小置信度：前提发生时，结论至少有50%几率发生

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- II. 核心挖掘引擎 --------------------

def _extract_transactions_from_graph(graph_json_path: str) -> Tuple[List[Set[str]], str, str]:
    """将图数据转化为事务集，并记录中心节点。"""
    with open(graph_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stars = data.get("stars", [])
    x0 = stars[0].get("center", "x0")
    y0 = stars[1].get("center", "y0") if len(stars) > 1 else x0

    transactions = []
    for star in stars:
        current_items = set()

        # 1. 节点属性 -> 谓词 (e.g., node.attr="val")
        for node, attrs in star.get("node_attrs", {}).items():
            for attr, val in attrs.items():
                if attr not in ["name", "label", "text_span", "center"]:
                    current_items.add(f'{attr}("{node}", "{val}")')

        # 2. 边关系 -> 谓词 (e.g., Relationship("type"))
        for rel, props in star.get("edges_attrs", {}).items():
            current_items.add(f'Relationship("{rel}")')

        if current_items:
            transactions.append(current_items)

    return transactions, x0, y0

def mine_rico_rules_locally(transactions: List[Set[str]], x0: str, y0: str) -> List[Dict]:
    """基于频繁项集挖掘生成 RICO 风格规则。"""
    all_items = [item for t in transactions for item in t]
    item_counts = Counter(all_items)

    # 筛选满足支持度的项
    frequent_items = [item for item, count in item_counts.items() if count >= MIN_SUPPORT]

    mined_rules = []
    # 挖掘二元关系 P -> Q (可根据需要扩展到多元组合)
    for p, q in combinations(frequent_items, 2):
        # 计算共现支持度
        support_both = sum(1 for t in transactions if p in t and q in t)

        if support_both >= MIN_SUPPORT:
            # 计算双向置信度
            conf_p_to_q = support_both / item_counts[p]
            conf_q_to_p = support_both / item_counts[q]

            if conf_p_to_q >= MIN_CONFIDENCE:
                mined_rules.append({
                    "rico": f'Q["{x0}", "{y0}"]({p} → {q})',
                    "support": support_both,
                    "confidence": round(conf_p_to_q, 4)
                })

            if conf_q_to_p >= MIN_CONFIDENCE:
                mined_rules.append({
                    "rico": f'Q["{x0}", "{y0}"]({q} → {p})',
                    "support": support_both,
                    "confidence": round(conf_q_to_p, 4)
                })

    return sorted(mined_rules, key=lambda x: (x['support'], x['confidence']), reverse=True)

# -------------------- III. 异步并行任务处理 --------------------

async def process_mining_task(file_path: str):
    """并行处理单个文件的规则挖掘。"""
    base_name = os.path.basename(file_path).replace('.json', '')
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_mined_rules.txt")

    # 使用线程池执行同步挖掘逻辑，避免阻塞事件循环
    loop = asyncio.get_running_loop()

    # 1. 提取事务
    transactions, x0, y0 = await loop.run_in_executor(None, _extract_transactions_from_graph, file_path)

    # 2. 挖掘规则
    start_time = time.perf_counter()
    rules = await loop.run_in_executor(None, mine_rico_rules_locally, transactions, x0, y0)
    duration = time.perf_counter() - start_time

    # 3. 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"### Mining Metadata ###\n")
        f.write(f"Source: {os.path.basename(file_path)}\n")
        f.write(f"x0: \"{x0}\", y0: \"{y0}\"\n")
        f.write(f"Duration: {duration:.4f}s\n\n")

        f.write("### Mined RICO Rules (Support-based) ###\n")
        for r in rules:
            f.write(f"Rule: {r['rico']}\n")
            f.write(f"Support: {r['support']} | Confidence: {r['confidence']}\n\n")

    print(f"✅ [SUCCESS] {base_name}: Mined {len(rules)} rules in {duration:.4f}s")

# -------------------- IV. 主流程 --------------------

async def main():
    graph_files = glob(os.path.join(INPUT_DIR, "*.json"))
    if not graph_files:
        print("❌ 未发现输入文件。")
        return

    print(f"--- 🚀 启动基于支持度的图挖掘 (阈值: {MIN_SUPPORT}) ---")
    start_wall = time.perf_counter()

    # 创建所有挖掘任务
    tasks = [process_mining_task(f) for f in graph_files]
    await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_wall
    print("\n" + "="*50)
    print("           ✨ 图规则挖掘任务完成 ✨")
    print("="*50)
    print(f"📋 处理文件数: {len(graph_files)}")
    print(f"⏳ 总耗时: {total_time:.2f} 秒")
    print(f"🚀 平均效率: {len(graph_files)/total_time:.2f} 文件/秒")
    print(f"📂 结果输出至: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())