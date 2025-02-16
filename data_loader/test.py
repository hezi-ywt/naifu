import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import io
from PIL import Image
# 假设您有一个 Arrow 文件 'data.arrow'
table = pa.ipc.open_file('/mnt/g/hf/danbooru_newest-webp-4Mpixel_pose_arrow/images/0000/00000.arrow').read_all()

# # 打印表格内容
# print(table)

# 将 Arrow 表转换为 Pandas DataFrame（如果需要）
df = table.to_pandas()
# print(df)


temp =df['image'].iloc[0]
temp2 = df['condition_image'].iloc[0]

image_bytes = io.BytesIO(temp)
image_bytes.seek(0)

image_bytes2 = io.BytesIO(temp2)
image_bytes2.seek(0)

# convert(RGB) has two purposes:
# 1. Convert the image to RGB mode. Some images are in grayscale/RGBA mode, which will cause channel
#    inconsistency in following processing.
# 2. Convert the image to RGB mode. Some images are in P mode, which will be forced to use NEAREST resample
#    method in resize (even if you specify LANCZOS), which will cause blurry images.
pil_image = Image.open(image_bytes).convert("RGB")
pil_image2 = Image.open(image_bytes2).convert("RGB")

print(pil_image.size)

#show
pil_image.save('/home/ywt/lab/naifu/data_loader/test_image.jpg')
pil_image2.save('/home/ywt/lab/naifu/data_loader/test_image2.jpg')