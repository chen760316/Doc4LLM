# master.py
import os
import sys
import time
import pickle
import paramiko
import threading
from multiprocessing.pool import ThreadPool
from src.kg_gen import KGGen
from src.kg_gen.models import Graph
from tests.utils.visualize_kg import visualize
from collections import defaultdict

# 🔧 集群信息（每个 host 有多个 slot）
REMOTE_NODES = [
    {"host": "192.168.1.101", "user": "ubuntu", "slots": 2},
    {"host": "192.168.1.102", "user": "ubuntu", "slots": 2},
]

REMOTE_WORK_DIR = "~/kg_remote/"
LOCAL_TEXT_DIR = "./texts/"
LOCAL_OUT_DIR = "./output/"

os.makedirs(LOCAL_OUT_DIR, exist_ok=True)


def run_remote_job(node, text_file, output_file):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(node["host"], username=node["user"])

    remote_input = os.path.join(REMOTE_WORK_DIR, os.path.basename(text_file))
    remote_output = os.path.join(REMOTE_WORK_DIR, os.path.basename(output_file))

    # 上传文件
    sftp = ssh.open_sftp()
    sftp.put(text_file, remote_input)
    sftp.put("worker.py", REMOTE_WORK_DIR + "worker.py")
    sftp.close()

    # 运行图生成任务
    cmd = f"cd {REMOTE_WORK_DIR} && python3 worker.py {remote_input} {remote_output}"
    print(f"[{node['host']}] Running: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.recv_exit_status()

    # 下载结果
    sftp = ssh.open_sftp()
    sftp.get(remote_output, output_file)
    sftp.close()
    ssh.close()


def load_graphs(output_dir):
    graphs = []
    for file in os.listdir(output_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(output_dir, file), "rb") as f:
                graphs.append(pickle.load(f))
    return graphs


def distribute_and_run():
    tasks = [f for f in os.listdir(LOCAL_TEXT_DIR) if f.endswith(".txt")]
    assigned = []

    idx = 0
    for node in REMOTE_NODES:
        for _ in range(node["slots"]):
            if idx >= len(tasks): break
            assigned.append((node, tasks[idx]))
            idx += 1

    def run(task):
        node, fname = task
        run_remote_job(node, LOCAL_TEXT_DIR + fname, LOCAL_OUT_DIR + fname.replace(".txt", ".pkl"))

    with ThreadPool(len(assigned)) as pool:
        pool.map(run, assigned)


if __name__ == "__main__":
    distribute_and_run()

    # 聚合
    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))
    graphs = load_graphs(LOCAL_OUT_DIR)
    combined_graph = kg.aggregate(graphs)

    # 匹配实体并连边
    all_entities = [g.entities for g in graphs]
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            for e1 in graphs[i].entities:
                for e2 in graphs[j].entities:
                    if e1 == e2:
                        combined_graph.relations.add((e1, "matched_entity", e2))
                        combined_graph.edges.add("matched_entity")

    # 增强（取前5高频实体）
    entity_degrees = defaultdict(int)
    for h, _, t in combined_graph.relations:
        entity_degrees[h] += 1
        entity_degrees[t] += 1
    top_entities = sorted(combined_graph.entities, key=lambda x: -entity_degrees.get(x, 0))[:5]

    # 可视化
    visualize(combined_graph, "final_enriched_kg.html")
    print("图谱已完成并保存为 final_enriched_kg.html")
