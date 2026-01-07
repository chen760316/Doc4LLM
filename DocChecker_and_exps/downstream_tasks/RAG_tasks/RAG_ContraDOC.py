"""
使用的数据集：ContraDOC
使用的基准：检索器模型：'all-MiniLM-L6-v2',生成器模型：'facebook/opt-125m'
"""
import json
import torch
import random
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # 引入通用生成模型
from tqdm import tqdm

# --- 配置文件 ---
# 假设您的 ContraDoc 数据集文件路径
CONTRADOC_FILE_PATH = "../../../datasets/data_from_related_work/ContraDOC/ContraDoc.json"
# 检索器模型 (用于生成文档嵌入)
RETRIEVER_MODEL = 'all-MiniLM-L6-v2'
# 生成器模型 (用于通用文本生成和矛盾核实)
# ⚠️ 注意：这里使用一个轻量级模型进行演示。实际任务应使用 Llama 3, Mistral, 或 T5/Flan 等大型生成模型。
GENERATOR_MODEL = 'facebook/opt-125m' # 推荐切换
TOP_K_RETRIEVAL = 2  # 检索 Top 2 最相关的段落
NUM_SAMPLES = 5  # 仅处理数据集中的少量样本进行演示

# --- 在所有函数调用之前定义 device ---
if torch.cuda.is_available():
    device = "cuda"
    # 显式使用 GPU 0
    if torch.cuda.device_count() > 0:
        device = "cuda:0"
else:
    device = "cpu"

print(f"Device set to: {device}")


# --- 1. 数据加载和预处理 (针对 ContraDoc 结构) ---

def load_and_prepare_contradoc(file_path: str, num_samples: int) -> List[Dict[str, Any]]:
    """加载 ContraDoc 数据集并将其结构化。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{file_path}'。请检查路径。")
        return []

    processed_data = []

    # 假设 ContraDoc 结构为 {"pos": {DOC_ID: {...}}, "neg": {DOC_ID: {...}}}
    # 我们关注有矛盾的 POSITIVE 样本
    positive_docs = list(data.get('pos', {}).items())

    # 随机采样并限制数量
    sampled_docs = random.sample(positive_docs, min(num_samples, len(positive_docs)))

    for doc_id, doc_data in sampled_docs:
        document_text = doc_data.get('text', '')
        # 矛盾证据：作为我们的 RAG 查询
        contradictory_query = doc_data.get('evidence', '')

        # 将长文档分割成段落/块 (Chunking) 作为知识库
        # 这里使用简单的换行符分割作为演示
        chunks = [c.strip() for c in document_text.split('\n\n') if c.strip()]

        # 确保有查询和上下文
        if contradictory_query and chunks:
            processed_data.append({
                'doc_id': doc_id,
                'query': contradictory_query,
                'document_chunks': chunks,
                # 原始文档包含的矛盾句和与之冲突的句（仅用于评估参考）
                'contradiction_evidence': contradictory_query,
                'reference_sentences': doc_data.get('ref sentences', ['N/A'])
            })

    print(f"✅ 成功加载并处理 {len(processed_data)} 个 ContraDoc 样本。")
    return processed_data

# --- 2. RAG 核心函数 (使用 .generate() 修正) ---

def run_rag_pipeline(qa_samples: List[Dict[str, Any]], retriever: SentenceTransformer, model: Any, tokenizer: AutoTokenizer):
    """
    运行 RAG 流程：检索、提示构建和生成。
    """
    # ... (提取 all_chunks 和 chunk_embeddings 的代码保持不变) ...
    all_chunks = []
    chunk_map = {}
    for sample in qa_samples:
        for chunk in sample['document_chunks']:
            if chunk not in chunk_map:
                chunk_map[chunk] = True
                all_chunks.append(chunk)

    # ... (生成嵌入的代码保持不变) ...
    print(f"\n🧠 正在对 {len(all_chunks)} 个上下文块生成嵌入...")
    chunk_embeddings = retriever.encode(all_chunks, convert_to_tensor=True, show_progress_bar=False)

    # ⚠️ 获取模型的最大输入长度
    max_model_length = model.config.max_position_embeddings
    max_new_tokens = 50
    # 留出 max_new_tokens 给生成的答案
    max_input_length = max_model_length - max_new_tokens

    # 确保 tokenizer 设置了 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # OPT 模型通用设置

    rag_results = []

    print(f"⏳ 正在执行 RAG 流程 (最大输入长度: {max_input_length} tokens)...")
    for sample in tqdm(qa_samples, desc="RAG Processing"):
        query = sample['query']

        # --- 2.1 检索 (Retrieval) ---
        query_embedding = retriever.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

        top_k_indices = torch.argsort(cos_scores, descending=True)[:TOP_K_RETRIEVAL]
        retrieved_chunks = [all_chunks[i] for i in top_k_indices]
        # retrieved_scores = [cos_scores[i].item() for i in top_k_indices] # 不需要用于生成

        # --- 2.2 提示构建 (Prompt Construction) ---
        context_text = "\n\n".join(retrieved_chunks)

        prompt_template = (
            "Given the following context from a document, verify if the claim below is true or false. "
            "If the context contains conflicting information, state the conflict.\n\n"
            f"Context:\n---\n{context_text}\n---\n\n"
            f"Claim to Verify: \"{query}\"\n\n"
            "Verification Output:"
        )

        # ⚠️ 关键修复：直接使用 tokenizer 编码和截断
        encoded_input = tokenizer(
            prompt_template,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True
        ).to(model.device)  # 将输入张量移动到 GPU

        # --- 2.3 生成 (Generation) ---
        try:
            output_sequences = model.generate(
                **encoded_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,  # 使用贪心解码
                pad_token_id=tokenizer.pad_token_id
            )

            # 解码结果并去除输入提示
            generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

            # 查找并去除输入的 prompt 部分
            prompt_len = encoded_input['input_ids'].shape[1]
            generated_token_ids = output_sequences[0][prompt_len:]
            generated_answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()


        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                generated_answer = "ERROR: CUDA Out of Memory during generation."
            else:
                generated_answer = f"ERROR: Device-side assert failed. Internal error."

        # --- 2.4 结果存储 ---
        rag_results.append({
            'doc_id': sample['doc_id'],
            'query_contradictory_claim': query,
            'reference_contradicting_sentences': sample['reference_sentences'],
            'retrieved_context': retrieved_chunks,
            'rag_verification_output': generated_answer
        })

    return rag_results

# --- 3. 主执行块 ---

if __name__ == '__main__':
    # 确保安装了 transformers 的 AutoModelForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ... (前面的配置和数据加载不变) ...
    # ⚠️ 强烈建议更换生成器模型，以获得稳定且有意义的矛盾核实结果
    GENERATOR_MODEL = 'facebook/opt-125m'  # 保持为 opt-125m

    # 1. 加载并准备数据
    qa_data = load_and_prepare_contradoc(CONTRADOC_FILE_PATH, NUM_SAMPLES)

    if not qa_data:
        exit()

    # 2. 模型加载 (更新为加载 Tokenizer 和 Model)
    print(f"\n🧠 正在加载检索器模型: {RETRIEVER_MODEL}")
    retriever_model = SentenceTransformer(RETRIEVER_MODEL, device=device)

    print(f"🧠 正在加载生成器模型: {GENERATOR_MODEL}")

    # 确保在加载模型时指定 device_map='auto' 或 .to(device)
    generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    generator_model = AutoModelForCausalLM.from_pretrained(GENERATOR_MODEL).to(device)

    # 3. 运行 RAG 流程 (传递 model 和 tokenizer)
    final_results = run_rag_pipeline(qa_data, retriever_model, generator_model, generator_tokenizer)

    # 4. 结果展示
    print("\n" + "=" * 80)
    print(f"ContraDoc RAG 矛盾核实模拟结果 ({len(final_results)} 样本)")
    print("=" * 80)

    for i, res in enumerate(final_results):
        print(f"\n--- 样本 {i + 1} (Doc ID: {res['doc_id']}) ---")
        print(f"❓ Query (矛盾句): {res['query_contradictory_claim'][:70]}...")
        print(f"💡 原始冲突句: {res['reference_contradicting_sentences'][0][:70]}...")
        print(f"🤖 RAG 核实结果:\n   {res['rag_verification_output']}")
        print(f"🔍 检索到的上下文数: {len(res['retrieved_context'])}")

    # 5. 保存结果到 JSON 文件
    output_filename = "contradoc_rag_simulation_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 详细 RAG 结果已保存到：{output_filename}")