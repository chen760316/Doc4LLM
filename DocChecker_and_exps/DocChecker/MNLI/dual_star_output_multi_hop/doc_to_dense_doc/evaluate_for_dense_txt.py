import json
import os
import re
import asyncio
from glob import glob
from dotenv import load_dotenv
from knowledge_graph_maker import OpenAIClient
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# === 初始化环境 ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

DENSE_TEXT_PATH = "dense_text.txt"
RULES_DIR = "dual_star/llm_rules_dense_txt"
EVAL_OUTPUT_DIR = "dual_star/evaluation_results"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

llm = OpenAIClient(
    model="gpt-5.2",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


async def verify_rule_semantics(rule: str, reference: Dict) -> Tuple[bool, str]:
    """返回 (是否有效, LLM的具体理由)"""
    system_prompt = "You are a logic expert. Reply ONLY in JSON format with two keys: 'valid' (boolean) and 'reason' (string)."

    user_prompt = f"""
Reference Case Reason: {reference.get('reason')}
Rule: {rule}

Task: Does this rule logically represent the contradiction described? 
Respond as: {{"valid": true/false, "reason": "..."}}
"""
    try:
        response = await asyncio.to_thread(
            llm.generate,
            user_message=user_prompt,
            system_message=system_prompt
        )
        # 简单解析 JSON 响应
        data = json.loads(re.search(r'\{.*\}', response, re.DOTALL).group())
        return data.get("valid", False), data.get("reason", "No reason provided.")
    except Exception:
        return False, "Evaluation failed."


def extract_rules_from_file(file_path: str) -> List[str]:
    """从 txt 文件中提取 Rules 部分的每一行"""
    rules = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 找到 Rules: 之后的部分
            if "Rules:" in content:
                rules_section = content.split("Rules:")[1].strip()
                rules = [line.strip() for line in rules_section.splitlines() if "Q[" in line]
    except Exception as e:
        print(f"读取文件错误 {file_path}: {e}")
    return rules


def get_center_from_rule(rule: str) -> str:
    """提取 Q[Center, ...] 中的第一个项"""
    match = re.search(r"Q\[(.*?),", rule)
    if match:
        return match.group(1).strip()
    return ""


async def evaluate_all_rules():
    # 加载数据
    with open(DENSE_TEXT_PATH, "r", encoding="utf-8") as f:
        dense_data = json.load(f)

    rule_files = glob(os.path.join(RULES_DIR, "*.txt"))
    total_items = len(dense_data)

    detailed_reports = []  # 详细记录
    covered_indices = set()

    sem = asyncio.Semaphore(15)  # 适当提高并发
    print(f"🚀 开始评估 {len(rule_files)} 个规则文件...")

    async def process_file(file_path):
        fname = os.path.basename(file_path)
        rules = extract_rules_from_file(file_path)
        file_results = []

        for rule in rules:
            center = get_center_from_rule(rule)
            matched_ref = None
            ref_idx = -1

            # 查找对应的原始矛盾
            for idx, item in enumerate(dense_data):
                if center.lower() in item['sentence_1'].lower() or center.lower() in item['sentence_2'].lower():
                    matched_ref = item
                    ref_idx = idx
                    break

            if matched_ref:
                async with sem:
                    is_valid, eval_reason = await verify_rule_semantics(rule, matched_ref)

                if is_valid:
                    covered_indices.add(ref_idx)

                file_results.append({
                    "rule": rule,
                    "ref_idx": ref_idx,
                    "is_valid": is_valid,
                    "eval_reason": eval_reason
                })

        return {"filename": fname, "results": file_results}

    # 执行异步任务
    tasks = [process_file(f) for f in rule_files]
    detailed_reports = await asyncio.gather(*tasks)

    # -------------------- 3. 保存结果 --------------------

    # 1. 保存详细报告 (JSON)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_log_path = os.path.join(EVAL_OUTPUT_DIR, f"detailed_eval_{timestamp}.json")
    with open(detailed_log_path, "w", encoding="utf-8") as f:
        json.dump(detailed_reports, f, indent=4, ensure_ascii=False)

    # 2. 保存统计摘要 (TXT)
    coverage_rate = (len(covered_indices) / total_items) * 100 if total_items > 0 else 0
    summary = (
        f"Evaluation Summary ({timestamp})\n"
        f"{'=' * 30}\n"
        f"Total Contradiction Items: {total_items}\n"
        f"Covered Items (Success):   {len(covered_indices)}\n"
        f"Logical Coverage Rate:     {coverage_rate:.2f}%\n"
        f"Processed Files:           {len(rule_files)}\n"
        f"{'=' * 30}\n"
    )

    summary_path = os.path.join(EVAL_OUTPUT_DIR, "summary.txt")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"\n✅ 评估完成！")
    print(f"📊 覆盖率: {coverage_rate:.2f}%")
    print(f"📄 详细日志已保存至: {detailed_log_path}")
    print(f"📜 摘要已更新至: {summary_path}")


if __name__ == "__main__":
    asyncio.run(evaluate_all_rules())