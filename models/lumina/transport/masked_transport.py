import torch as th
import numpy as np
from .transport import Transport, ModelType
from .utils import mean_flat

class MaskedTransport(Transport):
    """
    扩展Transport类，支持蒙版内外区域分别计算损失
    """
    
    def masked_training_losses(self, model, x1, mask=None, mask_weight=0.8, model_kwargs=None):
        """
        和原来计算差不多
        """
        # 如果没有提供蒙版，则使用原始的训练损失计算方法
        if mask is None:
            return self.training_losses(model, x1, model_kwargs)
            
        if model_kwargs is None:
            model_kwargs = {}
            
        # 生成采样和计划
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        
        # 处理条件输入
        if "cond" in model_kwargs:
            conds = model_kwargs.pop("cond")
            xt = [th.cat([x, cond], dim=0) if cond is not None else x for x, cond in zip(xt, conds)]
        
        # 模型前向传播
        model_output = model(xt, t, **model_kwargs)
        B = len(x0)
        
        terms = {}
        
        if self.model_type == ModelType.VELOCITY:
            if isinstance(x1, (list, tuple)):
                # 处理列表或元组输入的情况
                assert len(model_output) == len(ut) == len(x1) == len(mask), "输入数据、模型输出、速度场和蒙版的数量必须一致"
                
                inside_losses = []
                outside_losses = []
                total_losses = []
                
                for i in range(B):
                    assert (model_output[i].shape == ut[i].shape == x1[i].shape == mask[i].shape), \
                        f"形状不匹配: {model_output[i].shape}, {ut[i].shape}, {x1[i].shape}, {mask[i].shape}"
                    
                    # 计算像素级误差
                    pixel_errors = (ut[i] - model_output[i]) ** 2
                    
                    # 计算蒙版内部 loss
                    inside_mask = mask[i]
                    inside_loss = (pixel_errors * inside_mask).sum() / (inside_mask.sum() + 1e-8)
                    inside_losses.append(inside_loss)
                    
                    # 计算蒙版外部 loss
                    outside_mask = 1.0 - inside_mask
                    outside_loss = (pixel_errors * outside_mask).sum() / (outside_mask.sum() + 1e-8)
                    outside_losses.append(outside_loss)
                    
                    # 计算加权总 loss
                    total_loss = mask_weight * inside_loss + (1 - mask_weight) * outside_loss
                    total_losses.append(total_loss)
                
                # 将各部分 loss 堆叠为张量
                terms["inside_loss"] = th.stack(inside_losses, dim=0)
                terms["outside_loss"] = th.stack(outside_losses, dim=0)
                terms["task_loss"] = th.stack(total_losses, dim=0)
            else:
                # 处理单一张量输入的情况
                assert mask.shape == x1.shape, f"蒙版和输入形状不匹配: {mask.shape}, {x1.shape}"
                
                # 计算像素级误差
                pixel_errors = (model_output - ut) ** 2
                
                # 计算蒙版内部 loss
                inside_mask = mask
                inside_loss = (pixel_errors * inside_mask).sum(dim=list(range(1, len(mask.shape)))) / \
                              (inside_mask.sum(dim=list(range(1, len(mask.shape)))) + 1e-8)
                
                # 计算蒙版外部 loss
                outside_mask = 1.0 - inside_mask
                outside_loss = (pixel_errors * outside_mask).sum(dim=list(range(1, len(mask.shape)))) / \
                               (outside_mask.sum(dim=list(range(1, len(mask.shape)))) + 1e-8)
                
                # 计算加权总 loss
                total_loss = mask_weight * inside_loss + (1 - mask_weight) * outside_loss
                
                terms["inside_loss"] = inside_loss
                terms["outside_loss"] = outside_loss
                terms["task_loss"] = total_loss
        else:
            raise NotImplementedError("目前仅支持 VELOCITY 类型的模型")
        
        # 设置最终 loss 和其他参数
        terms["loss"] = terms["task_loss"]
        terms["task_loss"] = terms["task_loss"].clone().detach()
        terms["t"] = t
        
        return terms


def create_masked_transport(*args, **kwargs):
    """创建带蒙版功能的Transport对象
    
    用法与 create_transport 相同，返回 MaskedTransport 对象
    """
    from . import create_transport
    
    # 创建基础 Transport 对象
    transport = create_transport(*args, **kwargs)
    
    # 创建 MaskedTransport 对象，继承所有属性
    masked_transport = MaskedTransport(
        model_type=transport.model_type,
        path_type=transport.path_sampler.__class__.__name__,
        loss_type=transport.loss_type,
        train_eps=transport.train_eps,
        sample_eps=transport.sample_eps,
        snr_type=transport.snr_type,
        do_shift=transport.do_shift,
        seq_len=transport.seq_len
    )
    
    # 确保路径采样器被正确设置
    masked_transport.path_sampler = transport.path_sampler
    
    return masked_transport
