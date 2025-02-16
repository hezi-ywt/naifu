import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到路径

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from modules.lumina2_model import Lumina2Model

from lightning.fabric import Fabric

fabric = Fabric()



# 创建一个简单的数据集
class SimpleDataset(Dataset):
    def __init__(self, size=(512, 512), device="cuda"):
        super().__init__()
        self.size = size
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return 10  # 为测试只返回10个样本
        
    def __getitem__(self, idx):
        # 创建一个随机图像
        random_image = torch.rand(
            3, self.size[0], self.size[1], 
            device=self.device
        )
        # 创建一个简单的提示文本
        prompt = "a simple test prompt"
        
        return {
            "image": random_image,
            "prompt": prompt
        }

# 创建数据模块
class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = None
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = SimpleDataset()
        
    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        return None  # 不需要验证集
        
    def test_dataloader(self):
        return None  # 不需要测试集
        
    def predict_dataloader(self):
        return None  # 不需要预测集

def main():
    # 加载配置文件
    config = OmegaConf.load('/root/autodl-tmp/naifu/config/train_lumina2_test.yaml')
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建保存目录
    os.makedirs(config.trainer.checkpoint_dir, exist_ok=True)
    
    # 创建数据集和数据加载器
    train_dataset = SimpleDataset(device=device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 创建模型
    model_path = "/root/autodl-tmp/Lumina-Image-2.0"
    model = Lumina2Model(config, device, model_path)
    
    # 创建回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.trainer.checkpoint_dir,
        filename="{epoch}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss"
    )
    
    # 创建训练器
    # trainer = Trainer(
    #     max_steps=config.trainer.max_steps,
    #     max_epochs=config.trainer.max_epochs,
    #     accelerator="auto",
    #     devices=1,
    #     precision="bf16-mixed",
    #     callbacks=[checkpoint_callback],
    #     accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    #     gradient_clip_val=config.trainer.gradient_clip_val,
    # )
    
    from lumina2_fabric import MyCustomTrainer
    trainer = MyCustomTrainer(        
        accelerator=config.lightning.get("accelerator", "auto"),
        strategy=config.lightning.get("strategy", "auto"),
        devices=config.lightning.get("devices", "auto"),
        precision=config.lightning.get("precision", "bf16-mixed"),
        plugins=config.lightning.get("plugins", None),
        callbacks=config.lightning.get("callbacks", None),
        loggers=config.lightning.get("loggers", None),
        max_epochs=config.trainer.get("max_epochs", 1000),
        max_steps=config.trainer.get("max_steps", None),
        grad_accum_steps=config.trainer.get("accumulate_grad_batches", 1),
        limit_train_batches=config.trainer.get("limit_train_batches", float("inf")),
        limit_val_batches=config.trainer.get("limit_val_batches",  float("inf")),
        checkpoint_dir=config.trainer.get("checkpoint_dir", "./checkpoints"),
        checkpoint_frequency=config.trainer.get("checkpoint_frequency", 1),
    )
    # 开始训练
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()