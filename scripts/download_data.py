import os
from huggingface_hub import snapshot_download

# 配置参数
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def download_lcsts_data():
    """
    下载 LCSTS 数据集到 data 目录
    """
    repo_id = "hugcyp/LCSTS" 
    
    # 数据放在 data 文件夹
    local_dir = os.path.join(project_root, "data", "LCSTS_origin") 
    
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print(f"正在下载数据集: {repo_id} ...")
    print(f"目标目录: {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=['*.jsonl'],
            resume_download=True
        )
        print("\n数据集下载成功！")

    except Exception as e:
        print(f"\n下载失败: {e}")

if __name__ == "__main__":
    download_lcsts_data()