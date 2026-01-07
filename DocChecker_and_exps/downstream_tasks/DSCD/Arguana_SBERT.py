"""
使用的数据集：Arguana
使用的基准：'sentence-transformers/multi-qa-mpnet-base-dot-v1' # 用于检索任务
效果不好，未找到确定的反驳证据

📜 Arguana 的原始任务：反驳论证检索
目标：给定一个论证（Argument）作为查询（Query），系统需要在大型语料库（Corpus）中检索出最相关的、但持有相反立场的反驳论证（Counter-argument）。
任务类型：这是一个典型的信息检索 (Information Retrieval, IR) 任务，其特殊之处在于它要求检索模型理解**立场（Stance）和语义关联性（Semantic Relevance）**之间的复杂关系。
数据结构：
Queries：包含用户提供的论证（Argument）。
Corpus：包含大量潜在的反驳论证和无关文档。
qrels（或评分文件）：将每个 Query 映射到语料库中最合适的 Counter-argument，通常评分越高代表越好的反驳。

💻 适用于原始任务的代码实现（使用 Sentence Transformer）
要实现反驳论证检索，最先进且高效的方法是使用密集检索模型，例如 Sentence Transformer 系列模型，通过计算 Query 论证和 Corpus 文档的嵌入向量的相似度来进行排名。

矛盾检测（Contradiction Detection）或在 Arguana 语境下的反驳论证检索（Counter-argument Retrieval）需要模型理解两个独立的维度：
语义相关性 (Relevance)：查询和文档是否讨论同一个主题。
立场对立性 (Stance Opposition)：文档的观点是否与查询的观点相反。
要真正体现矛盾检测/反驳论证检索，代码需要在检索阶段后引入或集成**立场（Stance）**判断：
（理想情况 - 训练）：在 Sentence Transformer 模型上进行微调（例如，使用 Triplet Loss 或 Contrastic Loss），使其在相似度计算时能够将反驳论证拉近，同时将支持论证和无关论证推远。
（实际应用 - 混合模型）：
第一步：检索（如您代码所示）：使用相似度检索找到所有相关的文档（例如 Top 50）。
第二步：矛盾/立场分类：对这 Top 50 个相关文档和 Query 组成的对子，使用一个独立的二分类模型（例如 BERT 分类器）来判断立场关系（Contradictory/Supportive/Neutral）。
第三步：重排：只保留被分类为“Contradictory”的文档，并根据它们的相似度分数或分类器的置信度分数进行最终排名。
"""

# Arguana_Hybrid_Retrieval_Arrow.py

import json
import os
import torch
import random
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from tqdm import tqdm

# 需要安装 datasets 库
# pip install datasets

from datasets import load_from_disk, Dataset, DatasetDict

# 设置随机种子
random.seed(42)
torch.manual_seed(42)

# 定义输出文件名
ARGUANA_OUTPUT_FILE = "arguana_retrieval_results.json"
MAX_QUERIES = 50 # 限制采样的 Queries 数量

# --- 辅助保存函数 (保持不变) ---

def save_results_to_json(data: Dict[str, List[Dict[str, Any]]], file_path: str):
    """将检索结果字典保存为 JSON 文件。"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 检索结果已成功保存到文件: {file_path}")
    except Exception as e:
        print(f"❌ 保存结果时发生错误: {e}")


# --- 1. 数据加载函数 (修改为读取 Arrow 文件) ---

def load_arguana_data_from_arrow(arguana_base_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    从 Arguana 数据集文件夹中加载 Queries 和 Corpus 的 Arrow 文件，并转换为字典。

    Args:
        arguana_base_path: Arguana 数据集的基础目录路径。

    Returns:
        (queries_dict, corpus_dict)，其中键是文档ID，值是文本内容。
    """
    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}

    query_disk_path = os.path.join(arguana_base_path, 'queries')
    corpus_disk_path = os.path.join(arguana_base_path, 'corpus')

    print("🌍 正在从 Arguana 文件夹读取 Arrow 文件...")

    try:
        # 1. 加载 Queries Dataset
        query_dataset = load_from_disk(query_disk_path)
        print(f"✅ Queries Dataset 加载成功，共 {len(query_dataset)} 条记录。")
        # 假设 Query Dataset 包含 '_id' 和 'text' 列
        queries = {item['_id']: item['text'] for item in query_dataset}

        # 2. 加载 Corpus Dataset
        corpus_dataset = load_from_disk(corpus_disk_path)
        print(f"✅ Corpus Dataset 加载成功，共 {len(corpus_dataset)} 条记录。")
        # 假设 Corpus Dataset 包含 '_id' 和 'text' 列
        corpus = {item['_id']: item['text'] for item in corpus_dataset}

    except FileNotFoundError:
        print(f"❌ 错误: 找不到指定的 Arrow 文件夹路径。请检查 {arguana_base_path} 下的 'queries' 和 'corpus'。")
        raise
    except Exception as e:
        print(f"❌ 读取 Arrow 文件时发生错误: {e}")
        raise

    print(f"✅ 最终加载完成。Queries: {len(queries)}，Corpus: {len(corpus)}")
    return queries, corpus


# --- 2. 检索函数 (Sentence Transformer) (保持不变) ---
def step1_semantic_retrieval(queries: Dict[str, str], corpus: Dict[str, str], model: SentenceTransformer, top_k_retrieval: int = 10):
    # ... (函数体与上一回复保持一致) ...
    """
    第一步：使用 Sentence Transformer 检索 Top-K 个语义相似的文档。
    返回 Query-Corpus-Score 列表。
    """
    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())

    print(f"\n⏳ 正在计算 Corpus ({len(corpus_texts)} 个文档) 嵌入...")
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)

    retrieval_candidates = []

    print(f"\n⏳ 正在对 {len(queries)} 个 Query 进行检索...")
    for query_id, query_text in tqdm(queries.items(), desc="Step 1: Retrieving Candidates"):
        query_embedding = model.encode(query_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

        top_k_indices = torch.argsort(cos_scores, descending=True)[:top_k_retrieval]

        for idx in top_k_indices:
            doc_id = corpus_ids[idx]
            retrieval_candidates.append({
                "query_id": query_id,
                "query_text": query_text,
                "doc_id": doc_id,
                "doc_text": corpus[doc_id],
                "similarity_score": float(cos_scores[idx])
            })

    print(f"✅ Step 1 完成，共生成 {len(retrieval_candidates)} 个检索候选对。")
    return retrieval_candidates


# --- 3. 分类函数 (BERT Pipeline) - 最终修正 ---
def step2_stance_classification(candidates: List[Dict[str, Any]], classifier: Any) -> List[Dict[str, Any]]:
    """
    第二步：使用 NLI 模型对检索候选对进行矛盾/立场分类。
    修正了填充(padding)和标签映射。
    """
    print("\n⏳ 正在对候选对进行 Step 2: 立场/矛盾分类 (使用 NLI 模型)...")

    # 1. 输入格式：将 Query 和 Document 用 [SEP] 拼接，形成一个扁平的字符串列表
    input_texts = [
        f"{c['query_text']} [SEP] {c['doc_text']}"
        for c in candidates
    ]

    # 2. 关键修正：添加 padding=True 解决长度不一致的错误
    predictions = classifier(
        input_texts,
        padding=True,  # 启用填充
        truncation=True,
        max_length=512
    )

    classified_results = []

    # ⚠️ 打印前 5 个结果进行调试，检查实际标签
    print("\n🔎 调试信息：前 5 个分类结果的标签和分数：")
    for i, pred in enumerate(predictions[:5]):
        print(f"  - 候选 {i + 1} 标签: {pred['label']}, 分数: {pred['score']:.4f}")

    for candidate, pred in zip(candidates, predictions):
        # 3. 关键修正：使用 NLI 模型正确的字符串标签进行判断

        # 针对 'textattack/bert-base-uncased-mnli' 模型，通常输出的是字符串标签，
        # 并且 'CONTRADICTION' 是我们要找的。
        # 使用 .upper() 确保大小写不敏感。

        # 最终判断逻辑：
        is_contradictory = pred['label'].upper() == 'CONTRADICTION'

        # 如果模型输出是数字标签 (如 LABEL_0, LABEL_1, LABEL_2)，你需要根据打印结果修改此行：
        # is_contradictory = pred['label'].upper() == 'LABEL_0' # 如果打印结果显示 LABEL_0 是矛盾

        candidate['is_contradictory'] = is_contradictory
        candidate['classification_label'] = pred['label']
        candidate['classification_score'] = float(pred['score'])
        classified_results.append(candidate)

    print("✅ Step 2 完成。")
    return classified_results

# --- 4. 重排和输出函数 (保持不变) ---
def step3_rerank_and_output(classified_results: List[Dict[str, Any]], top_k_final: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    # ... (函数体与上一回复保持一致) ...
    final_results: Dict[str, List[Dict[str, Any]]] = {}

    for query_id in set(c['query_id'] for c in classified_results):
        contradictory_docs = [
            c for c in classified_results
            if c['is_contradictory']
        ]

        contradictory_docs.sort(key=lambda x: x['classification_score'], reverse=True)

        final_results[query_id] = contradictory_docs[:top_k_final]

    return final_results


# --- 5. 核心执行块 (替换分类模型) ---

if __name__ == '__main__':

    # 1. 模型加载
    RETRIEVAL_MODEL_NAME = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    print(f"🧠 正在加载检索模型: {RETRIEVAL_MODEL_NAME}")
    sbert_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)

    # 替换为公开可用的 MNLI 模型，例如 textattack 的版本
    CLASSIFIER_MODEL = 'textattack/bert-base-uncased-mnli'

    print(f"🧠 正在加载分类模型 (NLI 假设): {CLASSIFIER_MODEL}")
    # 必须使用 AutoModel 和 AutoTokenizer，因为 pipeline 默认的 text-classification
    # 可能无法正确处理 MNLI 的三个标签，但我们先尝试最简单的 pipeline:
    stance_classifier = pipeline("text-classification", model=CLASSIFIER_MODEL, device=0 if torch.cuda.is_available() else -1)    # 备注：即使是 NLI 模型，Hugging Face pipeline 可能会将其视为 sentiment-analysis
    # 或 text-classification，标签映射可能混乱。请确保 CLASSIFIER_MODEL 替换为您
    # 明确知道其标签映射的 NLI 模型。

    # 2. 数据加载
    ARGUANA_BASE_PATH = "../../../datasets/data_from_related_work/SPARSECL/Arguana"

    try:
        queries, corpus = load_arguana_data_from_arrow(ARGUANA_BASE_PATH)
    except Exception as e:
        print(f"程序因数据加载失败而终止: {e}")
        exit()

    # Queries 采样逻辑 (保持不变)
    if len(queries) > MAX_QUERIES:
        print(f"\n✂️ 正在从 {len(queries)} 条 Queries 中随机采样 {MAX_QUERIES} 条...")
        sampled_ids = random.sample(list(queries.keys()), MAX_QUERIES)
        sampled_queries = {id: queries[id] for id in sampled_ids}
        queries = sampled_queries
        print(f"✅ 采样完成。最终处理 Queries 数量: {len(queries)}")

    TOP_K_RETRIEVAL = 10
    TOP_K_FINAL = 2

    # 3. 流程执行
    candidates = step1_semantic_retrieval(queries, corpus, sbert_model, top_k_retrieval=TOP_K_RETRIEVAL)
    classified_candidates = step2_stance_classification(candidates, stance_classifier)
    final_retrieval = step3_rerank_and_output(classified_candidates, top_k_final=TOP_K_FINAL)

    # 4. 结果展示和保存 (保持不变)
    print("\n" + "=" * 80)
    print(f"🥇 混合模型反驳论证检索最终结果 (Top {TOP_K_FINAL})")
    print("=" * 80)

    for query_id, results in final_retrieval.items():
        print(f"\n[Query ID]: {query_id}")
        print(f"[Query Text]: {queries.get(query_id, 'N/A')[:70]}...")

        if not results:
            print("  --- 未找到确定的反驳论证。")
            continue

        for rank, res in enumerate(results):
            print(f"  --- Rank {rank + 1} (Contradictory) ---")
            print(f"    Doc ID: {res['doc_id']}")
            print(f"    Sim Score: {res['similarity_score']:.4f}")
            print(f"    Contradiction Score: {res['classification_score']:.4f}")
            print(f"    Snippet: {res['doc_text'][:50]}...")

    save_results_to_json(final_retrieval, ARGUANA_OUTPUT_FILE)

    print("\n--- 任务完成 ---")