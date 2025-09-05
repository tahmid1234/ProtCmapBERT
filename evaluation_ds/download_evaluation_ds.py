from huggingface_hub import hf_hub_download
import os

repo_name = "tahmid1234/ProtCmapBERT"
local_download_folder = "./evaluation_ds" # A folder in your current directory

# Make sure the folder exists
os.makedirs(local_download_folder, exist_ok=True)

# Download the mf.pt file to the specified folder
go_terms = hf_hub_download(
    repo_id=repo_name,
    filename="test_datatest_protein_data00.tfrecord",
    local_dir=local_download_folder+"/go_test",
)
go_terms = hf_hub_download(
    repo_id=repo_name,
    filename="test_datatest_protein_data01.tfrecord",
    local_dir=local_download_folder+"/go_test",
)
go_terms = hf_hub_download(
    repo_id=repo_name,
    filename="test_datatest_protein_data02.tfrecord",
    local_dir=local_download_folder+"/go_test",
)
go_terms = hf_hub_download(
    repo_id=repo_name,
    filename="test_datatest_protein_data03.tfrecord",
    local_dir=local_download_folder+"/go_test",
)

ec_number = hf_hub_download(
    repo_id=repo_name,
    filename="test_datatest_protein_data04.tfrecord",
    local_dir=local_download_folder+"/ec_test",
)
ec_number = hf_hub_download(
    repo_id=repo_name,
    filename="test_datatest_protein_data05.tfrecord",
    local_dir=local_download_folder+"/ec_test",
)

print(f"First four files contain GO Terms and remaining file are EC numbers")