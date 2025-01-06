python train_volcano.py --conf config/experiments/stage1_alignment.yaml
CUDA_VISIBLE_DEVICES=0 python train_volcano.py --conf config/experiments/stage3_instruct.yaml
torchrun --nproc_per_node=4 train_volcano.py --conf config/experiments/stage3_instruct.yaml