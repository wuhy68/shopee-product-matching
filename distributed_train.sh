#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
NUM_PROC=$1
PORT=${PORT:-29501}
shift
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port=$PORT train.py --kernel-type ecaresnet101d_512_10ep_lr4e-5 --image-size 512 --batch-size 8 --net-type ecaresnet101d --n-epochs 10 --fold 0 --init-lr 4e-5