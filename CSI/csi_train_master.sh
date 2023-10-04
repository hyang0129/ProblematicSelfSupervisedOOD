#!/bin/bash -l
#SBATCH -J csi_food
#SBATCH -t 3-00:00:00
#SBATCH -o ./outs/csi_food_train_final.o
#SBATCH -e ./outs/csi_food_train_final.e
#SBATCH -A cps -p tier3 -n 4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:a100:1

echo "Loading Packages"

#spack load cuda@10.2.89 /ydlu6td # cuda 11.8
#spack load gcc@8.2.0 /r54eocp # gcc 12.2

spack load /igzaycn # cuda 11.8
spack load /dr4ipev # gcc 12.2

source ../../miniconda/etc/profile.d/conda.sh
conda activate ptorch


CUDA_VISIBLE_DEVICES=0 python train_dk.py --dataset food \
                                        --model resnet18 \
                                        --mode simclr_CSI \
                                        --ood_mode ood_pre \
                                        --ood_dataset food \
                                        --ood_score CSI \
                                        --epochs 1000 \
                                        --shift_trans_type rotation \
                                        --print_score \
                                        --ood_samples 10 \
                                        --error_step 500 \
                                        --batch_size 512 \
                                        --runs 3 \
                                        --optimizer sgd
