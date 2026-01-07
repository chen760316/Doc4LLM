"""
弃用
基于NLU语义匹配的模型：判断上下文依赖或共现

| 模型名称                                         | 类型     | 用法备注              |
| -------------------------------------------- | ------ | ----------------- |
| `sentence-transformers/all-mpnet-base-v2`    | SBERT  | 精度高，适合通用语义匹配      |
| `princeton-nlp/sup-simcse-bert-base-uncased` | SimCSE | 更轻量，语义相关性强，适合匹配任务 |

不准确
根据两个句子中共同出现的单词判定依赖性和共现性
useless（仅写作）
"""

from sentence_transformers import SentenceTransformer, util

# 方式一
# 选择模型（可替换成 SimCSE）
# model_name = "sentence-transformers/all-mpnet-base-v2"  # SBERT
# # model_name = "princeton-nlp/sup-simcse-bert-base-uncased"  # SimCSE
#
# model = SentenceTransformer(model_name)

# 方式二
model = SentenceTransformer('/model/all-MiniLM-L6-v2')


def semantic_similarity_score(text_a: str, text_b: str) -> float:
    """
    计算两个文本的语义相似度 [0,1]
    """
    emb1 = model.encode(text_a, convert_to_tensor=True)
    emb2 = model.encode(text_b, convert_to_tensor=True)

    cosine_score = util.cos_sim(emb1, emb2)
    return cosine_score.item()


# 示例：两个事件之间的语义共现性
ex_a = "Tom was born in 1990"
ey_b = "Tom was dead in 1990"

score = semantic_similarity_score(ex_a, ey_b)
print(f"Semantic similarity score: {score:.4f}")
