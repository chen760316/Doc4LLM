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
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)
DENSE_TEXT_PATH = "dense_text.txt"
RULES_DIR = "dual_star/generate_rules_dense_txt"
EVAL_OUTPUT_DIR = "dual_star/generate_results"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)


async def verify_rule_semantics(rule: str, reference: Dict) -> Tuple[bool, str]:
    system_prompt = (
        "You are a logic validator. Your goal is to see if a symbolic rule correctly "
        "captures the ESSENCE of a text contradiction. "
        "Rules use x0 (for S1) and y0 (for S2). "
        "Output ONLY JSON: {'valid': bool, 'reason': str}."
    )
    user_prompt = f"""
        Reference Contradiction:
        S1: {reference.get('sentence_1')}
        S2: {reference.get('sentence_2')}
        Ground Truth Reason: {reference.get('reason')}

        Candidate Rule: {rule}

        Criteria:
        1. Does the rule correctly identify a conflict between x0 and y0 based on the reason?
        2. Are the predicates (attributes) logically plausible for the given text?
        If the rule structure matches the contradiction logic, mark valid=true.
        """
    try:
        response = await asyncio.to_thread(llm.generate, user_message=user_prompt, system_message=system_prompt)
        if not response: return False, "Empty response"
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match: return False, "No JSON found"
        data = json.loads(match.group())
        return data.get("valid", False), data.get("reason", "N/A")
    except Exception as e:
        return False, str(e)


def extract_rules_from_file(file_path: str) -> List[str]:
    rules = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "Rules:" in content:
                rules_section = content.split("Rules:")[1].strip()
                rules = [line.strip() for line in rules_section.splitlines() if "Q[" in line]
    except:
        pass
    return rules


def get_center_from_rule(rule: str) -> str:
    match = re.search(r"Q\[(.*?),", rule)
    return match.group(1).strip() if match else ""


# === 核心评估逻辑 (修复保存部分) ===

async def evaluate_all_rules():
    # 1. 加载数据
    with open(DENSE_TEXT_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    dense_data = all_data[:73]  # 保持你的 21 条切片
    total_items = len(dense_data)

    rule_files = glob(os.path.join(RULES_DIR, "*.txt"))
    covered_indices = set()
    sem = asyncio.Semaphore(10)  # 适当降低并发提高稳定性

    async def process_file(file_path):
        fname = os.path.basename(file_path)
        idx_match = re.search(r'case_(\d+)', fname)
        if not idx_match: return None

        ref_idx = int(idx_match.group(1))
        # 核心修复：如果索引超出了我们当前的评估切片范围，跳过
        if ref_idx >= total_items: return None

        target_case = dense_data[ref_idx]
        rules = extract_rules_from_file(file_path)

        file_results = []
        for rule in rules:
            # 核心改进：不再通过 get_center_from_rule 去做模糊文本匹配
            # 只要文件索引对了，我们就把这条规则发给 LLM 验证语义
            async with sem:
                is_valid, eval_reason = await verify_rule_semantics(rule, target_case)

            if is_valid:
                covered_indices.add(ref_idx)

            file_results.append({
                "rule": rule,
                "is_valid": is_valid,
                "reason": eval_reason
            })
        return {"filename": fname, "results": file_results}

    # 2. 执行任务并获取结果
    print(f"🚀 开始并发评估 {len(rule_files)} 个文件...")
    tasks = [process_file(f) for f in rule_files]
    detailed_reports = await asyncio.gather(*tasks)

    # --- 3. 新增：保存结果逻辑 ---

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细 JSON 报告
    detail_path = os.path.join(EVAL_OUTPUT_DIR, f"eval_detail_{timestamp}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detailed_reports, f, indent=4, ensure_ascii=False)

    # 计算并保存统计摘要
    coverage_rate = (len(covered_indices) / total_items) * 100 if total_items > 0 else 0
    summary_text = (
        f"--- Evaluation Summary ({timestamp}) ---\n"
        f"Total Target Items: {total_items}\n"
        f"Covered (Valid):    {len(covered_indices)}\n"
        f"Coverage Rate:      {coverage_rate:.2f}%\n"
        f"Files Processed:    {len(rule_files)}\n"
        f"Detailed Report:    {detail_path}\n"
        f"{'-' * 40}\n"
    )

    summary_file_path = os.path.join(EVAL_OUTPUT_DIR, "summary_history.txt")
    with open(summary_file_path, "a", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n✨ 评估保存成功！")
    print(f"📊 覆盖率: {coverage_rate:.2f}%")
    print(f"📂 详见目录: {EVAL_OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(evaluate_all_rules())