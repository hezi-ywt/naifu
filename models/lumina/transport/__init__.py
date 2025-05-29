from .transport import ModelType, PathType, Sampler, Transport, WeightType
from .masked_transport import MaskedTransport, create_masked_transport


def create_transport(
    path_type="Linear",
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    snr_type="uniform",
    do_shift=True,
    seq_len=1024,  # corresponding to 512x512
):
    """function for creating Transport object
    **Note**: model prediction defaults to velocity
    Args:
    - path_type: type of path to use; default to linear
    - learn_score: set model prediction to score
    - learn_noise: set model prediction to noise
    - velocity_weighted: weight loss by velocity weight
    - likelihood_weighted: weight loss by likelihood weight
    - train_eps: small epsilon for avoiding instability during training
    - sample_eps: small epsilon for avoiding instability during sampling
    """

    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }

    path_type = path_choice[path_type]

    if path_type in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else:  # velocity & [GVP, LINEAR] is stable everywhere
        train_eps = 0
        sample_eps = 0

    # create flow state
    state = Transport(
        model_type=model_type,
        path_type=path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        snr_type=snr_type,
        do_shift=do_shift,
        seq_len=seq_len,
    )

    return state


# 导出模块级函数，用于简化带蒙版的训练 loss 计算
def calc_masked_training_losses(transport, model, x1, mask=None, mask_weight=0.8, model_kwargs=None):
    """
    计算带蒙版的训练损失
    
    Args:
        transport: Transport 对象，或 MaskedTransport 对象
        model: 模型对象
        x1: 输入数据
        mask: 蒙版，如为 None 则按原始逻辑计算 loss
        mask_weight: 蒙版内部区域的权重，范围 [0,1]
        model_kwargs: 传递给模型的额外参数
        
    Returns:
        loss 术语字典
    """
    if isinstance(transport, MaskedTransport):
        # 如果已经是 MaskedTransport 实例，直接使用
        return transport.masked_training_losses(model, x1, mask, mask_weight, model_kwargs)
    elif mask is not None:
        # 如果提供了蒙版但 transport 不是 MaskedTransport 实例，创建一个
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
        masked_transport.path_sampler = transport.path_sampler
        return masked_transport.masked_training_losses(model, x1, mask, mask_weight, model_kwargs)
    else:
        # 如果没有蒙版，使用原始 training_losses
        return transport.training_losses(model, x1, model_kwargs)
