# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 --include localhost:2,3,4,5
# deepspeed --num_gpus=4 --master_port=9901 ./finetune_dsp.py
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=gpu --master_port=8802 ./finetune_dsp25.py