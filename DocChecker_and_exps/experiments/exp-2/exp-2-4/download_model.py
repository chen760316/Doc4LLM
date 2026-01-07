from huggingface_hub import snapshot_download
import os

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 执行下载
try:
    snapshot_download(
        repo_id="facebook/bart-base",
        local_dir="./bart-base",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("\n✅ 下载成功！模型已保存至 ./bart-base")
except Exception as e:
    print(f"\n❌ 下载失败: {e}")