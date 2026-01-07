import json
from sklearn.cluster import KMeans
import numpy as np

# 从 JSON 文件中读取数据（替换为你的文件路径）
with open('merged_enriched_graph.json', 'r') as file:
    merged_enriched_graph = json.load(file)

# 从原始文本文件读取
with open('../merged_text.txt', 'r', encoding='utf-8') as text_file:
    merged_text = text_file.read()

# 步骤 1: 计算每个条目的质心 (start + end) / 2
centroids = []
for entry in merged_enriched_graph:
    if "text_span" in entry["edge_properties"]:
        start = entry["edge_properties"]["text_span"].get("start", 0)
        end = entry["edge_properties"]["text_span"].get("end", 0)
        centroid = (start + end) / 2
        centroids.append(centroid)
    else:
        print(f"Warning: Missing 'text_span' in relationship for {entry['relationship']}")
        centroids.append(0)  # 默认质心为0

# 步骤 2: 使用 K-means 聚类将质心分成两类
kmeans = KMeans(n_clusters=2, random_state=42)
centroids = np.array(centroids).reshape(-1, 1)  # 将质心转换为二维数组
labels = kmeans.fit_predict(centroids)

# 步骤 3: 根据聚类结果分配条目到 enriched_graph1.json 或 enriched_graph2.json
enriched_graph1 = []
enriched_graph2 = []

for i, entry in enumerate(merged_enriched_graph):
    if labels[i] == 0:
        enriched_graph1.append(entry)
    else:
        enriched_graph2.append(entry)

# 步骤 4: 根据 text_span 从原始文本中提取对应文本并保存到 text1.txt 和 text2.txt
text1_lines = []
text2_lines = []

# 提取文本
for i, entry in enumerate(merged_enriched_graph):
    start = entry["edge_properties"]["text_span"].get("start", 0)
    end = entry["edge_properties"]["text_span"].get("end", 0)
    text_segment = merged_text[start:end]  # 从原文中提取对应的文本片段

    # 根据聚类标签决定分配到哪个文本文件
    if labels[i] == 0:
        text1_lines.append(text_segment)
    else:
        text2_lines.append(text_segment)

# 保存拆分后的 JSON 文件
with open("enriched_graph1.json", "w") as f1:
    json.dump(enriched_graph1, f1, indent=4)

with open("enriched_graph2.json", "w") as f2:
    json.dump(enriched_graph2, f2, indent=4)

# 保存提取出的文本
with open("text1.txt", "w", encoding='utf-8') as text1_file:
    text1_file.write("\n".join(text1_lines))

with open("text2.txt", "w", encoding='utf-8') as text2_file:
    text2_file.write("\n".join(text2_lines))

print("JSON files and text files have been successfully split.")
