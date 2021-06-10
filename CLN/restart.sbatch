#!/bin/sh

#SBATCH --job-name=v_2RRRRRR
#SBATCH --out=v_2RRRRRR.out
#SBATCH --account=pi-andrewferguson
#SBATCH --partition=andrewferguson-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu 
#SBATCH --cpus-per-task=10   # set this to the desired number of threads
#SBATCH --mem=17000

# LOAD MODULES
module load python 
conda activate /project/andrewferguson/Kirill/torch_kirill

# DO COMPUTE WORK
# srun /project/andrewferguson/Kirill/torch_kirill/bin/python pl_train_vae.py
srun /project/andrewferguson/Kirill/torch_kirill/bin/python pl_train_vae_CLNv1.py --v_num 2RRRRR
#srun /project/andrewferguson/Kirill/torch_kirill/bin/python pl_train_vae_CLN_scratch.py
