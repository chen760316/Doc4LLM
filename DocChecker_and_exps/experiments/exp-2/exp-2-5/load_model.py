# -*- coding: utf-8 -*-
import os
from huggingface_hub import snapshot_download

# 设置镜像环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def download_xlnet():
    repo_id = "xlnet-base-cased"
    local_dir = "./xlnet-base-cased-local"
    
    print(f"🚀 正在高速下载 {repo_id}...")
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 关键：直接下载文件而非链接，方便后续移动
        endpoint="https://hf-mirror.com",
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # 忽略其他框架权重，节省空间
    )
    
    print(f"\n✅ 下载完成！模型保存在: {local_dir}")

if __name__ == "__main__":
    download_xlnet()