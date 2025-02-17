import functools
from lightning.fabric.strategies import DeepSpeedStrategy



ds_strategy = DeepSpeedStrategy(
    stage=3,
    config={
        # "fp16": {
        #     "enabled": "auto",
        #     "loss_scale": 0,
        #     "loss_scale_window": 1000,
        #     "initial_scale_power": 16,
        #     "hysteresis": 2,
        #     "min_loss_scale": 1
        # },
        "fp16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "ignore_unused_parameters": True,
        },
        "zero_allow_untested_optimizer": True,
    }
)   


def _strategy():
    return ds_strategy


sdxl_ds_strategy = DeepSpeedStrategy(
    stage=2,
    config={
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "allgather_bucket_size": "auto",
            "reduce_scatter": True,
            "find_unused_parameters": True,
        },
        "gradient_clipping": 1,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False
    }
)   


def _sdxl_strategy():
    return sdxl_ds_strategy

import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch



ds_strategy1 = DeepSpeedStrategy(
    stage=1,
    config={
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 1,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "ignore_unused_parameters": True,
        },
        "zero_allow_untested_optimizer": True,
    }
)   

def _strategy1():
    return ds_strategy1

sdxl_ds_strategy1 = DeepSpeedStrategy(
    stage=1,
    config={
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 1,
            "overlap_comm": True,
        },
        "zero_allow_untested_optimizer": True,
    }
)   

def _sdxl_strategy1():
    return sdxl_ds_strategy1

sdxl_ds_strategy11 = DeepSpeedStrategy(
    stage=1,
    config={
        # "fp16": {
        #     "enabled": "auto",
        #     "loss_scale": 0,
        #     "loss_scale_window": 1000,
        #     "initial_scale_power": 16,
        #     "hysteresis": 2,
        #     "min_loss_scale": 1
        # },
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 1,
            "reduce_scatter": False,
            "overlap_comm": True,
        },
        "zero_allow_untested_optimizer": True,
    }
)   

def _sdxl_strategy11():
    return sdxl_ds_strategy11
