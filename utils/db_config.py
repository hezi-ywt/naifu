import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取连接信息
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_PORT")
MONGO_DB = os.getenv("MONGO_DB")

# 构建连接字符串
MONGO_URI = f"mongodb://{MONGO_USERNAME}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}"

# 使用连接
def get_mongo_client():
    from pymongo import MongoClient
    return MongoClient(MONGO_URI) 