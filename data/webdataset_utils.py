import webdataset as wds
import torch
from torchvision import transforms
from modules.cascade_model import EFFNET_PREPROCESS

def identity(x):
    return x

def create_webdataset(
    urls,
    batch_size,
    num_workers=2,
    shuffle_buffer=10000,
    handler=None,
    **kwargs
):
    """
    创建WebDataset数据加载器
    
    参数:
        urls: WebDataset数据源URL
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        shuffle_buffer: 随机打乱缓冲区大小
        handler: 自定义数据处理函数
    """
    
    if handler is None:
        handler = identity

    dataset = wds.WebDataset(urls, handler=handler)
    
    # 数据预处理转换
    preproc = transforms.Compose([
        *EFFNET_PREPROCESS.transforms
    ])

    dataset = dataset.decode("pil").to_tuple("jpg", "txt")
    
    def preprocess(sample):
        image, text = sample
        image = preproc(image)
        return {
            "pixels": image,
            "prompts": text,
            "is_latent": False
        }

    dataset = dataset.map(preprocess)
    
    # 启用随机打乱
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    
    # 设置批次大小
    dataset = dataset.batched(batch_size)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=2,
    ) 