#!/bin/bash

#python -m torch.distributed.launch --nproc_per_node 1 --master_port 23124 few_shot_prune.py \
#       --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 128 --dsets_type flowers102 \
#       --exp_name GradSP_Ablate05_Starting32_SEED0 \
#       --model_type swin_adapters  --finetune 6 --type_adapters series --size_adapters 32 \
#       --ssl_warmup_epochs 20 --total_epochs 80 --prune_type GradSP --prune_amount 0.5 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 1 --weighted 0 --scaling 1
#
#python -m torch.distributed.launch --nproc_per_node 1 --master_port 23124 few_shot_prune.py \
#       --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 128 --dsets_type flowers102 \
#       --exp_name GradSP_Ablate05_Starting64_SEED0 \
#       --model_type swin_adapters  --finetune 6 --type_adapters series --size_adapters 64 \
#       --ssl_warmup_epochs 20 --total_epochs 80 --prune_type GradSP --prune_amount 0.5 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 1 --weighted 0 --scaling 1
#
#python -m torch.distributed.launch --nproc_per_node 1 --master_port 23124 few_shot_prune.py \
#       --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 128 --dsets_type flowers102 \
#       --exp_name GradSP_Ablate05_Starting128_SEED0 \
#       --model_type swin_adapters  --finetune 6 --type_adapters series --size_adapters 128 \
#       --ssl_warmup_epochs 20 --total_epochs 80 --prune_type GradSP --prune_amount 0.5 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 1 --weighted 0 --scaling 1
#
#python -m torch.distributed.launch --nproc_per_node 1 --master_port 23124 few_shot_prune.py \
#       --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 128 --dsets_type flowers102 \
#       --exp_name GradSP_Ablate05_Starting256_SEED0 \
#       --model_type swin_adapters  --finetune 6 --type_adapters series --size_adapters 256 \
#       --ssl_warmup_epochs 20 --total_epochs 80 --prune_type GradSP --prune_amount 0.5 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 1 --weighted 0 --scaling 1

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 few_shot_prune.py \
       --cfg configs/swin_large_patch4_window7_224.yaml --batch-size 128 --dsets_type cifar-100 \
       --exp_name TINA_SwinLarge_Adapters_Size4_Pruning05_SEED0 \
       --model_type swin_adapters  --finetune 6 --type_adapters series --size_adapters 4 \
       --ssl_warmup_epochs 20 --total_epochs 80 --prune_type layerwise --prune_amount 0.5 \
       --prune_struct "structured" \
       --amp-opt-level O2 --seed 0 --range 4 --weighted 0 --scaling 1

##################################################################################################

#python -m torch.distributed.launch --nproc_per_node 1 --master_port 54321 few_shot_prune.py \
#       --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 512 --exp_name TINA_Size4_0.9_SEED0 --dsets_type flowers102 \
#       --model_type swin_adapters  --finetune 6 --type_adapters series --size_adapters 4 \
#       --ssl_warmup_epochs 20 --total_epochs 100 --prune_type layerwise --prune_amount 0.9 \
#       --amp-opt-level O2 --seed 0 --range 4 --scaling 1 --weighted 0 --prune_struct "structured"

#python -m torch.distributed.launch --nproc_per_node 1 --master_port 54321 few_shot_prune.py \
#       --cfg configs/vit_base_16_224.yaml --batch-size 512 --dsets_type flowers102 \
#       --exp_name TINA_Size4_0.9_SEED0 \
#       --model_type vit_adapters  --finetune 6 --type_adapters series --size_adapters 4 \
#       --ssl_warmup_epochs 20 --total_epochs 100 --prune_type layerwise --prune_amount 0.9 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 4 --weighted 0 --scaling 1


#python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 few_shot_prune.py \
#       --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 512 --dsets_type cifar-10 \
#       --exp_name TINA_Size4_0.9_SEED0 \
#       --model_type swin_adapters  --finetune 6 --type_adapters series --size_adapters 4 \
#       --ssl_warmup_epochs 20 --total_epochs 100 --prune_type layerwise --prune_amount 0.9 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 4 --weighted 0 --scaling 1

#python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 few_shot_prune.py \
#       --cfg configs/vit_base_16_224.yaml --batch-size 512 --dsets_type cifar-10 \
#       --exp_name TINA_Size4_0.9_SEED0 \
#       --model_type vit_adapters  --finetune 6 --type_adapters series --size_adapters 4 \
#       --ssl_warmup_epochs 20 --total_epochs 100 --prune_type layerwise --prune_amount 0.9 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 4 --weighted 0 --scaling 1

#################################################################

#python -m torch.distributed.launch --nproc_per_node 1 --master_port 14523 few_shot_prune.py \
#       --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 512 --dsets_type flowers102 \
#       --exp_name TINA_AdaptersLast_Size4_SEED0 \
#       --model_type swin_adapters_layer  --finetune 6 --type_adapters series --size_adapters 4 \
#       --ssl_warmup_epochs 20 --total_epochs 100 --prune_type layerwise --prune_amount 0.9 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 3
#
#python -m torch.distributed.launch --nproc_per_node 1 --master_port 14523 few_shot_prune.py \
#       --cfg configs/swin_tiny_patch4_window7_224.yaml --batch-size 512 --dsets_type cifar-10 \
#       --exp_name TINA_AdaptersLast_Size4_SEED0 \
#       --model_type swin_adapters_layer  --finetune 6 --type_adapters series --size_adapters 4 \
#       --ssl_warmup_epochs 20 --total_epochs 100 --prune_type layerwise --prune_amount 0.9 \
#       --prune_struct "structured" \
#       --amp-opt-level O2 --seed 0 --range 3