import os
import tarfile
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm
from hfutils.operate import upload_file_to_file, upload_directory_as_archive, upload_directory_as_directory




repo_id = 'heziiiii/lu2'

local_file = '/mnt/public/lu2/results_cosine_2e-4_bs64_infsss/checkpoint-e12_s23973/consolidated.00-of-01.pth'
file_in_repo = 'results_cosine_2e-4_bs64_infssss/checkpoint-e12_s23973/consolidated.00-of-01.pth'
upload_file_to_file(
    local_file=local_file,
    repo_id=repo_id,
    file_in_repo=file_in_repo,
    repo_type="model",
    hf_token=os.environ.get("HF_TOKEN")
)