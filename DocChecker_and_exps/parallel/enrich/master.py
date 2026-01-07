#!/usr/bin/env python3
import os, pickle
import paramiko
from multiprocessing.pool import ThreadPool
from src.kg_gen import KGGen
from tests.utils.visualize_kg import visualize
from collections import defaultdict

# 你的虚拟机配置
NODES = [
    {"host": "192.168.56.101", "user": "ubuntu"},
    {"host": "192.168.56.102", "user": "ubuntu"},
]
WORKERS_PER_NODE = 1  # 每台 VM 启动多少 worker 进程

LOCAL_TEXTS = "texts/"
REMOTE_WORKDIR = "/home/ubuntu/kg_work/"
LOCAL_RESULTS = "results/"

os.makedirs(LOCAL_RESULTS, exist_ok=True)

def run_one(node, text_file):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(node["host"], username=node["user"])
    sftp = ssh.open_sftp()
    sftp.mkdir(REMOTE_WORKDIR, exist_ok=True)
    remote_in = REMOTE_WORKDIR + os.path.basename(text_file)
    remote_out = remote_in.replace(".txt", ".pkl")
    sftp.put(text_file, remote_in)
    sftp.put("worker.py", REMOTE_WORKDIR + "worker.py")
    sftp.close()

    cmd = f"cd {REMOTE_WORKDIR} && python3 worker.py {remote_in} {remote_out}"
    stdin, stdout, stderr = ssh.exec_command(cmd)
    stdout.channel.recv_exit_status()

    local_out = LOCAL_RESULTS + os.path.basename(remote_out)
    sftp = ssh.open_sftp()
    sftp.get(remote_out, local_out)
    sftp.close()
    ssh.close()
    print(f"✔ Fetched {local_out}")

def main():
    # 分配任务：每文本分配给一个子节点执行
    text_files = sorted(os.listdir(LOCAL_TEXTS))
    tasks = []
    for i, tf in enumerate(text_files):
        node = NODES[i % len(NODES)]
        tasks.append((node, LOCAL_TEXTS + tf))

    with ThreadPool(len(tasks)) as pool:
        pool.starmap(run_one, tasks)

    # 汇聚所有子图
    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))
    graphs = []
    for fn in os.listdir(LOCAL_RESULTS):
        if fn.endswith(".pkl"):
            with open(LOCAL_RESULTS + fn, "rb") as f:
                graphs.append(pickle.load(f))
    combined = kg.aggregate(graphs)

    # 合并 matched_entity
    all_entities = [g.entities for g in graphs]
    for i in range(len(graphs)):
        for j in range(i+1, len(graphs)):
            for e in graphs[i].entities & graphs[j].entities:
                combined.relations.add((e, "matched_entity", e))
                combined.edges.add("matched_entity")

    visualize(combined, "final_enriched_kg.html")
    print("✅ 完成可视化：final_enriched_kg.html")

if __name__ == "__main__":
    main()
