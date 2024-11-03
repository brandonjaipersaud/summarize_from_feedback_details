#!/bin/bash
#SBATCH --job-name=rlhf   
#SBATCH --partition=high_priority 
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --requeue
#SBATCH --exclusive



# module load cuda/12.2
export WANDB_TAGS=single-run,no-tag-$(git rev-parse --short HEAD)

# Single model and seed configuration
MODEL="EleutherAI/pythia-2.8b-deduped"  # Choose your model
SEED=44413  # Choose your seed
LR=3e-6

# Set model paths
REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED
POLICY_MODEL_PATH=models/$MODEL/policy_model_$SEED

# Set batch sizes for pythia-2.8b
local_rollout_forward_batch_size=32
gradient_accumulation_steps=16
local_micro_batch_size=4
local_eval_batch_size=1

echo "Running training with MODEL: $MODEL and SEED: $SEED"

srun poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/ppo.py \
    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=$POLICY_MODEL_PATH \
    --push_to_hub \
    --seed=$SEED