from huggingface_hub import HfApi, HfFolder, Repository
import os

# 设置你自己的访问令牌
token = "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"

# 登录 Hugging Face 账户
HfFolder.save_token(token)

# 设置模型名称和本地路径
model_name = "Laxhar/noob_openpose"
local_dir = "/data/cn/openpose"

# 登录 Hugging Face 账户
HfFolder.save_token(token)

# 初始化 HfApi 实例
api = HfApi()

# 如果模型/存储库不存在，则创建它
try:
    api.create_repo(repo_id=model_name, exist_ok=True)
except Exception as e:
    print(f"Failed to create the repository: {e}")
    pass

# 上传本地文件夹
api.upload_folder(
    folder_path=local_dir,
    path_in_repo="",
    repo_id=model_name,
    repo_type="model"
)

print(f"Model folder pushed to https://huggingface.co/{model_name}")

