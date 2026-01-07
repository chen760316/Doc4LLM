import json
import os
import re
import time
import asyncio
from glob import glob
from collections import Counter
from itertools import combinations
from functools import partial
from typing import List, Dict, Any, Tuple, Set

# -------------------- I. 配置与环境 --------------------

INPUT_DIR = "dual_star_json_with_attrs"
OUTPUT_DIR = "dual_star/mined_rules_v1"  # 存放挖掘出的规则
MIN_SUPPORT = 2  # 最小支持度阈值：至少在2个星型结构中出现

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- II. 核心挖掘引擎 (同步计算) --------------------

def extract_transactional_data(graph_json_path: str) -> List[Set[str]]:
    """
    将图数据转化为事务数据集。每个 Star 被视为一个 Transaction，
    包含的所有属性和关系被视为 Items。
    """
    with open(graph_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transactions = []
    for star in data.get("stars", []):
        current_items = set()

        # 1. 提取节点属性谓词 (例如: type="Person")
        for node_name, attrs in star.get("node_attrs", {}).items():
            for attr, val in attrs.items():
                if attr not in ["name", "label", "text_span", "center"]:
                    # 简化谓词格式便于挖掘
                    current_items.add(f'{attr}("{val}")')

        # 2. 提取关系谓词
        for rel_name, props in star.get("edges_attrs", {}).items():
            # rel_name 通常包含关系类型
            current_items.add(f'HasRelation("{rel_name}")')

        if current_items:
            transactions.append(current_items)
    return transactions


def mine_frequent_rules(transactions: List[Set[str]], min_support: int) -> List[Dict]:
    """
    基于支持度挖掘 RICO 风格规则。
    """
    # 1. 统计所有项的频率
    all_items = [item for trans in transactions for item in trans]
    item_counts = Counter(all_items)

    # 2. 筛选频繁项 (满足最小支持度)
    frequent_items = [item for item, count in item_counts.items() if count >= min_support]

    rules = []
    # 3. 挖掘二元规则: P -> Q (可扩展为多元)
    for p, q in combinations(frequent_items, 2):
        # 计算共现次数 (Support)
        support_count = sum(1 for trans in transactions if p in trans and q in trans)

        if support_count >= min_support:
            # 计算置信度 (Confidence) = Support(P,Q) / Support(P)
            confidence_p_to_q = support_count / item_counts[p]
            confidence_q_to_p = support_count / item_counts[q]

            # 生成 RICO 格式
            if confidence_p_to_q >= 0.5:  # 设定最小置信度过滤
                rules.append({
                    "rule": f"Q[x, y]({p} → {q})",
                    "support": support_count,
                    "confidence": round(confidence_p_to_q, 4)
                })
            if confidence_q_to_p >= 0.5:
                rules.append({
                    "rule": f"Q[x, y]({q} → {p})",
                    "support": support_count,
                    "confidence": round(confidence_q_to_p, 4)
                })

    # 按支持度排序
    return sorted(rules, key=lambda x: x['support'], reverse=True)


# -------------------- III. 异步任务包装器 --------------------

async def process_single_file_mining(file_path: str):
    """
    处理单个文件的挖掘任务。
    """
    base_name = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_DIR, base_name.replace('.json', '_mined.txt'))

    loop = asyncio.get_running_loop()

    # 1. 提取数据
    transactions = await loop.run_in_executor(None, extract_transactional_data, file_path)

    # 2. 挖掘规则
    start_time = time.perf_counter()
    rules = await loop.run_in_executor(None, partial(mine_frequent_rules, transactions, MIN_SUPPORT))
    duration = time.perf_counter() - start_time

    # 3. 写入结果
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"--- Support-based Rule Mining Report ---\n")
        f.write(f"Source: {base_name}\n")
        f.write(f"Mined Rules: {len(rules)}\n")
        f.write(f"Mining Latency: {duration:.4f}s\n")
        f.write("-" * 40 + "\n\n")

        for r in rules:
            f.write(f"Rule: {r['rule']}\n")
            f.write(f"Support: {r['support']} | Confidence: {r['confidence']}\n\n")

    print(f"✅ [DONE] {base_name} (Mined {len(rules)} rules in {duration:.2f}s)")


# -------------------- IV. 主运行函数 --------------------

async def main():
    graph_files = glob(os.path.join(INPUT_DIR, "*.json"))

    if not graph_files:
        print(f"❌ 未找到文件于 {INPUT_DIR}")
        return

    print(f"--- 🚀 启动图挖掘流程 (支持度阈值: {MIN_SUPPORT}) ---")
    start_wall = time.perf_counter()

    # 并发执行本地计算任务
    tasks = [process_single_file_mining(f) for f in graph_files]
    await asyncio.gather(*tasks)

    end_wall = time.perf_counter()
    print("\n" + "=" * 50)
    print(f"🏁 挖掘任务全部完成")
    print(f"⏱️ 总耗时: {end_wall - start_wall:.2f} 秒")
    print(f"📂 结果目录: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())