import os
from huggingface_hub import snapshot_download

# path settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def download_lcsts_data():
    """
    download LCSTS dataset to data/
    """
    repo_id = "hugcyp/LCSTS" 
    
    # Data is placed in the data folder
    local_dir = os.path.join(project_root, "data", "LCSTS_origin") 
    
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    print(f"Downloading dataset: {repo_id} ...")
    print(f"Target directory: {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=['*.jsonl'],
            resume_download=True
        )
        print("\nDataset downloaded successfully!")

    except Exception as e:
        print(f"\nDownload failed: {e}")

if __name__ == "__main__":
    download_lcsts_data()