from huggingface_hub import hf_hub_download
import os

repo_name = "tahmid1234/ProtCmapBERT"
local_download_folder = "./all_models/ProtCmapBERT" # A folder in your current directory

# Make sure the folder exists
os.makedirs(local_download_folder, exist_ok=True)

# Download the model files to the specified folder
mf_file_path = hf_hub_download(
    repo_id=repo_name,
    filename="dr_01_mf_lr_7e-06_cmap_bias_per_head_alpha_clipping_1_.pt",
    local_dir=local_download_folder,
)


bp_file_path = hf_hub_download(
    repo_id=repo_name,
    filename="dr_01_bp_lr_7e-06_cmap_bias_per_head_alpha_clipping_1_.pt",
    local_dir=local_download_folder,
)

cc_file_path = hf_hub_download(
    repo_id=repo_name,
    filename="dr_01_cc_lr_7e-06_cmap_bias_per_head_alpha_clipping_1_.pt",
    local_dir=local_download_folder,
)

bp_file_path = hf_hub_download(
    repo_id=repo_name,
    filename="dr_01_ec_lr_7e-06_cmap_bias_per_head_alpha_clipping_1_.pt",
    local_dir=local_download_folder,
)

print("Four models are successfully downloaded")