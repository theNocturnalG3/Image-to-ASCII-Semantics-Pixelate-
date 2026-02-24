import os
from huggingface_hub import snapshot_download

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root, "models")
    local_dir = os.path.join(models_dir, "segformer-b0-ade-512-512")
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

    print("Model downloaded to:", local_dir)

if __name__ == "__main__":
    main()