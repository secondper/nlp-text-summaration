import os
import shutil
from huggingface_hub import snapshot_download

def download_lcsts_data():
    """
    下载 LCSTS 数据集到 data 目录
    """
    # ================= 配置区域 =================
    # 你指定的那个数据集链接 ID
    repo_id = "hugcyp/LCSTS" 
    
    # 按照作业目录规范，数据应该放在 data 文件夹
    local_dir = "./data/LCSTS_origin" 
    
    # 国内镜像加速（如果网络不好请取消注释）
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # ===========================================

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

        # 自动清理 .cache 垃圾文件
        cache_path = os.path.join(local_dir, ".cache")
        if os.path.exists(cache_path):
            print("正在清理 .cache 缓存...")
            shutil.rmtree(cache_path)
            print("清理完成。")

    except Exception as e:
        print(f"\n下载失败: {e}")

if __name__ == "__main__":
    download_lcsts_data()