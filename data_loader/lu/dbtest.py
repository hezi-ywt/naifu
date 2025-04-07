import motor.motor_asyncio
import asyncio
# client = pymongo.MongoClient('mongodb://nieta:.nietanieta.@8.153.97.53:27815/')
# db = client['danbooru']
# collection = db['captions']

# # cursor = collection.find()


# mongo_data = collection.find_one({"_id": int(1)})
# print(mongo_data)
# print(type(mongo_data))
# # for doc in cursor:
#     # print(doc)
#     # break


async def find_by_id_async(id, mongo_uri="mongodb://nieta:.nietanieta.@8.153.97.53:27815/"):
    # 确保id是数字类型
    numeric_id = int(id)
    # 计算gemini_captions_danbooru的集合名
    gemini_collection_name = numeric_id // 100000
    # 连接MongoDB
    client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
    # 这里明确指定数据库名称，而不是使用get_database()
    db = client["danbooru"]  # 使用danbooru数据库
    # 并行查询三个集合
    captions_future = db["captions"].find_one({"_id": numeric_id})
    pics_future = db["pics"].find_one({"_id": numeric_id})
    
    # 获取gemini数据库并查询
    gemini_db = client["gemini_captions_danbooru"]
    gemini_future = gemini_db[f"{gemini_collection_name}"].find_one({"_id": numeric_id})
    
    # 等待所有查询完成
    captions_result = await captions_future
    pics_result = await pics_future
    gemini_result = await gemini_future
    # 关闭连接
    client.close()
    # 构建结果
    result = {}
    # 将captions_result的内容直接展开到结果中
    if captions_result:
        result.update(captions_result)
    # 将pics_result作为origin_danbooru_data
    # if pics_result:
    #     result["origin_danbooru_data"] = pics_result
    # 将gemini_result作为gemini_caption_v2
    if gemini_result:
        result["gemini_caption_v2"] = gemini_result["caption"]
    return result

# 创建主异步函数
async def main():
    result = await find_by_id_async(1)
    print(result)

# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())