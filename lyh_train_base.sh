#!/bin/bash
set -e

# 环境变量设置
export HF_LEROBOT_HOME=/opt/liblibai-models/user-workspace2/dataset
export CUDA_VISIBLE_DEVICES=6,7

# WandB 配置（可选，如果不设置会用默认值）
export WANDB_PROJECT=pi0_base_to_libero_training
export WANDB_ENTITY=lyh_5321-sjtu
export HF_LEROBOT_HOME=/opt/liblibai-models/user-workspace2/dataset
# 自定义 checkpoint 保存路径
CHECKPOINT_DIR="/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/pi0_lyh_libero_lora_official_full"

echo "================================================"
echo "训练配置："
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  Checkpoint 路径: $CHECKPOINT_DIR/lyh_libero_lora_v1"
echo "  WandB Project: $WANDB_PROJECT"
echo "================================================"

# 训练（使用 torchrun 多 GPU）
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
	  scripts/train_pytorch.py pi0_lyh_libero_lora_official_full \
	    --exp-name lyh_libero_lora_v1 \
	      --checkpoint-base-dir "$CHECKPOINT_DIR" \
	        --batch-size 32 \
		  --num-train-steps 50000 \
		    --save-interval 1000 \
		      --keep-period 5000 \
		        --wandb-enabled 

echo "================================================"
echo "训练完成！"
echo "Checkpoint 保存在: $CHECKPOINT_DIR/lyh_libero_lora_v1"
echo "================================================"
