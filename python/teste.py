from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="pyronear/pyro-sdis",
    repo_type="dataset",
    local_dir="pyro_sdis"
)

print("Download terminado!")
