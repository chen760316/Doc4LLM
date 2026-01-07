from knowledge_graph_maker import OpenAIClient
import json
from dotenv import load_dotenv
import os
import re
import difflib
import time
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed # 引入多线程模块

# --- 配置 ---
MAX_RETRIES = 3    # 最大重试次数
MAX_WORKERS = 10    # 线程池最大工作线程数 (可根据API限制和带宽调整)

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
INPUT_FILE = "sampled_data.json"
OUTPUT_PATH = "output_knowledge_graph_v3.json"


# --- 数据加载函数 (保持不变) ---
def load_contra_doc_data(file_path: str) -> Optional[Dict[str, Dict[str, Dict[str, Any]]]]:
    """尝试从 ContraDoc.json 文件中加载数据。"""
    if not os.path.exists(file_path):
        print(f"🚨 错误：输入文件未找到: {file_path}")
        return None

    print(f"--- 🚀 正在读取文件: {file_path} ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict) or 'pos' not in data or 'neg' not in data:
            print("🚨 错误：文件内容结构不符合预期的 ContraDoc 格式 (缺少 'pos' 或 'neg' 键)。")
            return None

        print(f"✅ 成功加载 ContraDoc 数据。")
        return data

    except json.JSONDecodeError as e:
        print(f"🚨 错误：JSON 解析失败。请检查文件格式是否正确。错误信息: {e}")
        return None
    except Exception as e:
        print(f"🚨 发生未知错误: {e}")
        return None

# --- 索引查找函数 (保持不变) ---
def find_text_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    """
    找到 substring 在 text 中的第一个出现位置，返回 (start, end) 索引。
    如果找不到，返回 None。
    """
    match = re.search(re.escape(substring), text)
    if match:
        return (match.start(), match.end())
    else:
        return None

def find_fuzzy_match_span(text: str, pattern: str, window: int = 100) -> Optional[Tuple[int, int]]:
    """模糊匹配关系文本的索引。"""
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
        name1 = edge["node_1"].get("name", "")
        span1 = find_text_span(original_text, name1)
        if span1:
            edge["node_1"]["text_span"] = {"start": span1[0], "end": span1[1]}

        name2 = edge["node_2"].get("name", "")
        span2 = find_text_span(original_text, name2)
        if span2:
            edge["node_2"]["text_span"] = {"start": span2[0], "end": span2[1]}

        relation = edge.get("relationship", "")
        if relation:
            span = find_fuzzy_match_span(original_text, relation)
            if span:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["text_span"] = {"start": span[0], "end": span[1]}

    return edges

# --- LLM 系统提示词 (保持不变) ---
system_prompt = (
    "You are an expert in extracting knowledge graphs from natural language text.\n"
    "Given an input text, extract all entities and relationships mentioned.\n"
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


# --- 核心处理函数 (适配并发) ---
def process_single_doc(
    doc_id: str,
    doc_data: Dict[str, Any],
    llm: OpenAIClient,
    system_prompt: str
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    处理单个文档，提取图谱并添加索引，加入重试机制和清洗逻辑。
    返回 (doc_id, 提取的边列表)。
    """

    original_text = doc_data.get("text", "")
    if not original_text:
        return doc_id, None

    print(f"\n--- 🛠️ 正在处理文档 ID: {doc_id} ---")

    for attempt in range(MAX_RETRIES):
        try:
            # 增加退避等待
            if attempt > 0:
                time.sleep(2 ** attempt)
                print(f"   重试 ({attempt + 1}/{MAX_RETRIES}): ID {doc_id}")

            response = llm.generate(user_message=original_text, system_message=system_prompt)

            # 🌟 增强的 JSON 清洗和提取逻辑
            json_text = response.strip()

            # 1. 移除 Markdown 围栏
            if json_text.startswith("```"):
                json_text = re.sub(r"```(json)?\s*", "", json_text, flags=re.DOTALL).strip()
                if json_text.endswith("```"):
                    json_text = json_text[:-3].strip()

            # 2. 尝试清理 JSON 错误
            cleaned_json_text = json_text.replace('\n', '').replace('\t', '')
            cleaned_json_text = re.sub(r',\s*([\]}])', r'\1', cleaned_json_text)

            # 3. 提取完整的 JSON 数组
            match = re.search(r"(\[.*\])", cleaned_json_text, re.DOTALL)

            if not match:
                raise ValueError(f"LLM 响应中未找到有效 JSON 数组。")

            # 4. 安全加载
            edges = json.loads(match.group(1))

            # 确保 edges 是一个列表
            if not isinstance(edges, list):
                 raise ValueError("LLM 响应解析后不是一个有效的 JSON 数组列表。")


            # 5. 添加索引信息
            edges_with_spans = add_text_spans_with_fuzzy_relation(edges, original_text)

            # 6. 添加文档ID等元数据
            doc_type = doc_data.get("doc_type", "N/A")
            for edge in edges_with_spans:
                if "edge_properties" not in edge:
                    edge["edge_properties"] = {}
                edge["edge_properties"]["doc_id"] = doc_id
                edge["edge_properties"]["doc_type"] = doc_type

            print(f"✅ 文档 {doc_id} 处理成功 (尝试 {attempt + 1})。")
            return doc_id, edges_with_spans

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                # print(f"⚠️ 警告: 文档 {doc_id} 格式错误 ({e})。正在重试... ({attempt + 1}/{MAX_RETRIES})")
                continue
            else:
                print(f"🚨 错误: 文档 {doc_id} 格式错误，重试 {MAX_RETRIES} 次后失败 ({e})。跳过该文档。")
                return doc_id, None
        except Exception as e:
            # 捕获其他未知错误（如 API 调用失败）
            print(f"🚨 错误: 处理文档 {doc_id} 时发生未知错误: {e}")
            return doc_id, None

    return doc_id, None

# --- 主执行逻辑 (修改为多线程) ---
def main():
    """主函数，使用 ThreadPoolExecutor 进行多线程处理。"""
    contra_doc_data = load_contra_doc_data(INPUT_FILE)
    if not contra_doc_data:
        return

    all_extracted_edges = []

    # 收集所有需要处理的文档列表
    tasks = []
    for doc_type in ['pos', 'neg']:
        if doc_type in contra_doc_data:
            for doc_id, doc_data in contra_doc_data[doc_type].items():
                # 确保 doc_data 中包含 doc_type 信息，方便后面元数据添加
                doc_data["doc_type"] = doc_type
                tasks.append((doc_id, doc_data))

    doc_count = len(tasks)
    successful_docs = 0
    failed_docs = 0
    processed_count = 0

    print(f"\n--- ⚙️ 启动线程池，总文档数: {doc_count}, 最大工作线程数: {MAX_WORKERS} ---")

    # 使用 ThreadPoolExecutor 进行并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_doc = {
            executor.submit(
                process_single_doc, doc_id, doc_data, llm, system_prompt
            ): doc_id
            for doc_id, doc_data in tasks
        }

        # 收集结果
        for future in as_completed(future_to_doc):
            doc_id = future_to_doc[future]
            processed_count += 1

            try:
                # future.result() 返回 (doc_id, edges)
                _, edges = future.result()
                if edges:
                    all_extracted_edges.extend(edges)
                    successful_docs += 1
                    print(f"✅ 文档 {doc_id} 提取成功。进度: {processed_count}/{doc_count}")
                else:
                    failed_docs += 1
                    print(f"➖ 文档 {doc_id} 提取失败。进度: {processed_count}/{doc_count}")

            except Exception as exc:
                print(f'🚨 文档 {doc_id} 任务生成了一个未捕获的异常: {exc}')
                failed_docs += 1 # 捕获异常也计入失败

    print(f"\n--- 🎉 处理完成。 ---")
    print(f"   - 总文档数: {doc_count}")
    print(f"   - 成功提取文档数: {successful_docs}")
    print(f"   - 失败文档数 (重试后跳过或异常): {failed_docs}")
    print(f"   - 总提取边数: {len(all_extracted_edges)}")

    # 打印格式化结果 (只打印前几个)
    print("\n[示例输出前 5 条边]:")
    print(json.dumps(all_extracted_edges[:5], indent=2, ensure_ascii=False))

    # 🌟 Save to JSON file
    # 保存到文件
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_extracted_edges, f, ensure_ascii=False, indent=2)

    print(f"\nKnowledge graph saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()