#!/bin/bash
#SBATCH -J hq
# SBATCH -o dsb-selfrdb-simplify-%j.log
# SBATCH -e dsb-selfrdb-simplify-%j.err
#SBATCH -N 1 -n 1
#SBATCH -w node02
#SBATCH --gres=gpu:1

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=smc-r07-07:29400 --standalone train_ct_us.py --method dsb --noiser self --gamma_type linear --network selfrdb --batch_size 16 --prior afhq-cat-384 --dataset afhq-dog-384 --val_prior afhq-val-cat-384 --val_data afhq-val-dog-384 --lr 1e-4 --epoch 10 --repeat_per_epoch 128 --use_amp --training_timesteps 10 --inference_timestep 10 --exp_name dsb-us2ct --skip_epochs 0 --simplify