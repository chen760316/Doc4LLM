"""
--- 🎉 处理完成。 ---
   - 总耗时: 99.19 秒 (约 1.65 分钟)
   - 总文档数: 10
   - 成功提取文档数: 8
   - 失败文档数: 2
   - 总提取边数: 77
   - 平均单条文档处理速度: 9.92 秒/篇
"""
from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
import difflib
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed # 引入多线程模块
import time # 1. 引入时间模块

# --- 配置 ---
MAX_RETRIES = 3 # 最大重试次数
MAX_WORKERS = 10 # 线程池最大工作线程数

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

# 升级到 gpt-4o-mini
llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9
)

# --- 配置 ---
INPUT_FILE = "sampled_data.json" # 假设输入文件名为 sampled_data.json

# --- 数据加载函数 (更新为新格式) ---
def load_new_format_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """尝试从新的 JSON 文件中加载数据（列表格式）。"""
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return None

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("🚨 错误：文件内容结构不符合预期的 JSON 列表格式。")
            return None

        print(f"✅ 成功加载 {len(data)} 条数据。")
        return data

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None

# --- 索引查找函数 (保持不变) ---
def find_text_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    """找到 substring 在 text 中的第一个出现位置，返回 (start, end) 索引。"""
    match = re.search(re.escape(substring), text)
    if match:
        return (match.start(), match.end())
    else:
        return None

def find_fuzzy_match_span(text: str, pattern: str, window: int = 100) -> Optional[Tuple[int, int]]:
    """模糊匹配关系文本的索引。"""
    # ... (函数体保持不变) ...
    text_len = len(text)
    pattern_len = len(pattern)
    best_ratio = 0
    best_span = (0, 0)

    if pattern_len == 0 or pattern_len > text_len:
        return None

    step = max(10, pattern_len // 2)
    search_window = max(pattern_len, window)

    for start in range(0, text_len, step):
        end = min(start + search_window, text_len)
        segment = text[start:end]
        ratio = difflib.SequenceMatcher(None, segment, pattern).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match_start = segment.find(pattern)
            if best_match_start != -1:
                 best_span = (start + best_match_start, start + best_match_start + pattern_len)
            else:
                 best_span = (start, end)
        if best_ratio > 0.95:
            break

    return best_span if best_ratio > 0 else None


def add_text_spans_with_fuzzy_relation(edges: List[Dict[str, Any]], original_text: str) -> List[Dict[str, Any]]:
    """为节点和关系添加在原始文本中的索引。"""
    for edge in edges:
        # 节点 1 索引
        name1 = edge["node_1"].get("name", "")
        span1 = find_text_span(original_text, name1)
        if span1:
            edge["node_1"]["text_span"] = {"start": span1[0], "end": span1[1]}

        # 节点 2 索引
        name2 = edge["node_2"].get("name", "")
        span2 = find_text_span(original_text, name2)
        if span2:
            edge["node_2"]["text_span"] = {"start": span2[0], "end": span2[1]}

        # 关系索引 (模糊匹配)
        relation = edge.get("relationship", "")
        if relation:
            span = find_fuzzy_match_span(original_text, relation)
            if span:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["text_span"] = {"start": span[0], "end": span[1]}

    return edges


system_prompt = (
    "You are an expert in extracting knowledge graphs from natural language text.\n"
    "Given an input text, extract all entities and relationships mentioned.\n"
    # ... (Prompt其余部分保持不变) ...
    "For each entity, include all relevant attributes found in the text—such as age, occupation, nationality, status, location, etc.\n"
    "For each relationship, include a natural language description and, if available, any attributes such as time, place, reason, or method.\n"
    "Return your output as a JSON array, following this flexible schema:\n"
    "[\n"
    "  {\n"
    '    "node_1": { "label": "<ontology_label>", "name": "<entity_name>", "...": "<other attributes if available>" },\n'
    '    "node_2": { "label": "<ontology_label>", "name": "<entity_name>", "...": "<other attributes if available>" },\n'
    '    "relationship": "<natural language description of the relation>",\n'
    '    "edge_properties": { "...": "<additional relationship attributes if available>" }\n'
    "  }\n"
    "]\n"
    "Respond with only the JSON array—no explanations, no markdown, no extra text."
    "IMPORTANT: Your response MUST STRICTLY be a JSON array. Do not include any text, markdown code fences (```json), or explanations outside of the array."
)


def process_single_doc(
    doc_index: int,
    doc_data: Dict[str, Any],
    llm: OpenAIClient,
    system_prompt: str
) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
    """处理单个文档，提取图谱并添加索引，加入重试机制和清洗逻辑。返回文档索引和提取的边。"""

    # 🌟 新数据格式：将 conflicting_texts 列表内容合并为原始文本
    conflicting_texts = doc_data.get("text", [])
    original_text = "\n".join(conflicting_texts).strip()

    if not original_text:
        print(f"\n--- ⚠️ 正在处理文档索引: {doc_index} ---")
        print(f"⚠️ 警告: 文档 {doc_index} 的 'conflicting_texts' 为空或缺失。跳过。")
        return doc_index, None

    print(f"\n--- 🛠️ 正在处理文档索引: {doc_index} (Text Len: {len(original_text)}) ---")

    for attempt in range(MAX_RETRIES):
        try:
            response = llm.generate(user_message=original_text, system_message=system_prompt)

            # 🌟 增强的 JSON 清洗和提取逻辑
            json_text = response.strip()

            # 1. 移除 Markdown 围栏 (```json)
            if json_text.startswith("```"):
                json_text = re.sub(r"```(json)?\s*", "", json_text, flags=re.DOTALL).strip()
                if json_text.endswith("```"):
                    json_text = json_text[:-3].strip()

            # 2. 尝试清理 JSON 错误 (移除换行、制表符，并修正多余逗号)
            cleaned_json_text = json_text.replace('\n', '').replace('\t', '')
            # 移除对象或数组结束前的多余逗号 (解决 Expecting ',' delimiter 的常见原因)
            cleaned_json_text = re.sub(r',\s*([\]}])', r'\1', cleaned_json_text)

            # 3. 提取完整的 JSON 数组
            match = re.search(r"(\[.*\])", cleaned_json_text, re.DOTALL)

            if not match:
                raise ValueError(f"LLM 响应中未找到有效 JSON 数组。尝试 {attempt + 1}/{MAX_RETRIES}")

            # 4. 安全加载
            edges = json.loads(match.group(1))

            # 5. 添加索引信息
            edges_with_spans = add_text_spans_with_fuzzy_relation(edges, original_text)

            # 6. 添加文档ID等元数据
            for edge in edges_with_spans:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                # 使用文档的索引和 main_entity 作为 ID
                edge["edge_properties"]["doc_index"] = doc_index
                edge["edge_properties"]["source_id"] = doc_data.get("main_entity", f"Index_{doc_index}")
                edge["edge_properties"]["source_domain"] = doc_data.get("source_domain", "Unknown")


            print(f"✅ 文档 {doc_index} 处理成功 (尝试 {attempt + 1})。")
            return doc_index, edges_with_spans

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                print(f"⚠️ 警告: 文档 {doc_index} 格式错误 ({e})。正在重试... ({attempt + 1}/{MAX_RETRIES})")
                continue
            else:
                print(f"🚨 错误: 文档 {doc_index} 格式错误，重试 {MAX_RETRIES} 次后失败 ({e})。跳过该文档。")
                return doc_index, None
        except Exception as e:
            print(f"🚨 错误: 处理文档 {doc_index} 时发生未知错误: {e}")
            return doc_index, None

    return doc_index, None


# --- 主执行逻辑 (修改为多线程) ---

def main():
    """主函数，使用 ThreadPoolExecutor 进行多线程处理，并包含计时模块。"""
    new_format_data = load_new_format_data(INPUT_FILE)
    if not new_format_data:
        return

    # --- 计时开始 ---
    overall_start_time = time.time()

    all_extracted_edges = []
    doc_count = len(new_format_data)
    successful_docs = 0
    failed_docs = 0

    print(f"\n--- ⚙️ 启动线程池，最大工作线程数: {MAX_WORKERS} ---")

    # 使用 ThreadPoolExecutor 进行并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_doc = {
            executor.submit(
                process_single_doc, i, doc_data, llm, system_prompt
            ): i
            for i, doc_data in enumerate(new_format_data)
        }

        # 收集结果
        for future in as_completed(future_to_doc):
            doc_index = future_to_doc[future]
            try:
                _, edges = future.result()
                if edges:
                    all_extracted_edges.extend(edges)
                    successful_docs += 1
                else:
                    failed_docs += 1
            except Exception as exc:
                print(f'🚨 文档 {doc_index} 任务生成了一个异常: {exc}')
                failed_docs += 1

    # --- 计时结束 ---
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time  # 计算总秒数

    print(f"\n--- 🎉 处理完成。 ---")
    print(f"   - 总耗时: {total_duration:.2f} 秒 (约 {total_duration / 60:.2f} 分钟)")
    print(f"   - 总文档数: {doc_count}")
    print(f"   - 成功提取文档数: {successful_docs}")
    print(f"   - 失败文档数: {failed_docs}")
    print(f"   - 总提取边数: {len(all_extracted_edges)}")

    # 计算效率指标
    if doc_count > 0:
        avg_time = total_duration / doc_count
        print(f"   - 平均单条文档处理速度: {avg_time:.2f} 秒/篇")

    # 保存结果
    output_path = "output_knowledge_graph_v3.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_extracted_edges, f, ensure_ascii=False, indent=2)

    print(f"\nKnowledge graph saved to: {output_path}")


if __name__ == "__main__":
    main()