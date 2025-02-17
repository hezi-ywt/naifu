import functools
from lightning.fabric.strategies import DeepSpeedStrategy



ds_strategy = DeepSpeedStrategy(
    stage=3,
    config={
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 1e9,
            "stage3_prefetch_bucket_size": 1e9,
            "stage3_param_persistence_threshold": 1e7,
            "stage3_max_live_parameters": 3e9,
            "stage3_max_reuse_distance": 3e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "ignore_unused_parameters": True,
            "round_robin_gradients": True
        },
        "gradient_clipping": 2.0,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 4,
        "train_batch_size": 32,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 100,
        "amp": {
            "enabled": False,
        },
        "communication_data_type": "bf16",
        "distributed": {
            "init_method": "env://",
            "nccl": {
                "debug": 1,
                "ib_timeout": 23,
                "local_rank": -1
            }
        },
        "aio": {
            "block_size": 2097152,
            "queue_depth": 16,
            "thread_count": 2,
            "single_submit": False,
            "overlap_events": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    }
)   





sdxl_ds_strategy = DeepSpeedStrategy(
    stage=3,
    config={
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "ignore_unused_parameters": True,
            "round_robin_gradients": True
        },
        "gradient_clipping": 2.0,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 8,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 100,
        "communication_data_type": "bf16",
        "amp": {
            "enabled": False,
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
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
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 1e9,
            "allgather_bucket_size": 1e9,
            "ignore_unused_parameters": True,
            "round_robin_gradients": True
        },
        "gradient_clipping": 2.0,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 4,
        "train_batch_size": 32,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 100,
        "communication_data_type": "bfloat16",
        "amp": {
            "enabled": False
        }
    }
)   

def _strategy1():
    return ds_strategy1

sdxl_ds_strategy1 = DeepSpeedStrategy(
    stage=1,
    config={
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            "ignore_unused_parameters": True,
            "round_robin_gradients": True
        },
        "gradient_clipping": 2.0,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": 8,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 100,
        "communication_data_type": "bfloat16",
        "amp": {
            "enabled": False
        }
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

def _strategy():
    return ds_strategy2

ds_strategy2 = DeepSpeedStrategy(
    stage=2,
    config={
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "ignore_unused_parameters": True
        },
        "gradient_clipping": 1,
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "zero_allow_untested_optimizer": True,
        "zero_force_ds_cpu_optimizer": False,
        "wall_clock_breakdown": False,
        "distributed": {
            "init_method": "env://",
            "find_unused_parameters": True,
            "gradient_as_bucket_view": True,
            "nccl": {
                "debug": 1,
                "ib_timeout": 23
            }
        },
        "aio": {
            "block_size": 2097152,
            "queue_depth": 16,
            "thread_count": 2,
            "single_submit": False,
            "overlap_events": True
        }
    }
)   

def _strategy2():
    return ds_strategy2

# SDXL 版本的 stage 2 配置
sdxl_ds_strategy2 = DeepSpeedStrategy(
    stage=2,
    config={
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 1e9,
            "allgather_bucket_size": 1e9,
            "round_robin_gradients": True,
            "prefetch_bucket_size": 1e9,
            "stage3_max_live_parameters": 3e9,
            "stage3_max_reuse_distance": 3e9,
            "ignore_unused_parameters": True
        },
        "gradient_clipping": 2.0,
        "gradient_accumulation_steps": 1,
        "train_micro_batch_size_per_gpu": 4,
        "train_batch_size": 32,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 100,
        "communication_data_type": "bf16",
        "distributed": {
            "init_method": "env://",
            "nccl": {
                "debug": 1,
                "ib_timeout": 23,
                "local_rank": -1
            }
        },
        "amp": {
            "enabled": False
        },
        "aio": {
            "block_size": 2097152,
            "queue_depth": 16,
            "thread_count": 2,
            "single_submit": False,
            "overlap_events": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    }
)

def _sdxl_strategy2():
    return sdxl_ds_strategy2
