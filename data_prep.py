from huggingface_hub import snapshot_download

# Basic download method
downloaded_file_path = snapshot_download(
    repo_id="Vision-CAIR/cc_sbu_align",
    repo_type="dataset",
    revision="main",
    # Optional: download to a specific local path
    local_dir="./finally/"
)