import os
from huggingface_hub import snapshot_download

# path settings
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def download_bart_model():
    """
    download BART model to checkpoint/
    """
    repo_id = "OpenMOSS-Team/bart-base-chinese" # model ID
    local_dir = os.path.join(project_root, "checkpoint") # save path

    print(f"Preparing to download model: {repo_id} ...")
    print(f"Target directory: {local_dir}")

    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
        print("\nModel downloaded successfully!")
        print(f"Files saved in: {local_dir}")
        print("Includes files: config.json, pytorch_model.bin, vocab.txt")
        
    except Exception as e:
        print(f"\nDownload failed: {e}")

if __name__ == "__main__":
    download_bart_model()