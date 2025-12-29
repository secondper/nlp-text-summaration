import os
from huggingface_hub import snapshot_download

def download_bart_model():
    """
    下载 LCSTS 数据集到 checkpoint 目录
    """
    repo_id = "OpenMOSS-Team/bart-base-chinese" # 模型 ID
    local_dir = "./checkpoint" # 保存路径

    print(f"正在准备下载模型: {repo_id} ...")
    print(f"目标目录: {os.path.abspath(local_dir)}")

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "config.json", 
                "pytorch_model.bin", 
                "vocab.txt"
            ]
        )
        print("\n模型下载成功！")
        print(f"文件已保存在: {local_dir}")
        print("包含文件: config.json, pytorch_model.bin, vocab.txt")
        
    except Exception as e:
        print(f"\n下载失败: {e}")

if __name__ == "__main__":
    download_bart_model()