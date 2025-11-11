from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contrastive_model.dataset import VideoTextDataset, data_collator
from contrastive_model.model import CustomModel
import torch
from dataclasses import dataclass
import logging
from tqdm import tqdm
import time
import os
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from youcook import CausalYoucookDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class Config:
    """
    Configuration class for the CLIP training script.
    """
    video_feature_dim: int = 3072 # 2048 appearance + 1024 flow
    embed_dim: int = 768  # Embedding dimension
    epochs: int = 8  # Number of training epochs
    batch_size: int = 16  # Batch size
    lr: float = 4e-4
    text_encoder_path: str = "dVAR/LLM_Weights/bert-base-uncased"  # Pretrained text encoder
    temperature: float = 0.5  # Temperature for loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    return args

def train():
    args = get_args()
    # ddp setup
    local_rank = int(os.environ['LOCAL_RANK']) 
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda:{}".format(local_rank))

    # 设置随机种子以确保可重复性
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    runner_unique_id = str(int(time.time()))
    save_dir = f"ckpts/clip_cook{runner_unique_id}"
    if dist.get_rank() == 0:
        print(f"Save dir: {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
    
        logging.basicConfig(
            filename=f"{save_dir}/train.log",
            level=logging.WARNING,
            format="%(message)s"
            # filemode="a"
        )

    # clip_dataset = VideoTextDataset(split="trainval", mode='trainval')
    clip_dataset = CausalYoucookDataset(split="training", mode='trainval')
   
    train_sampler = DistributedSampler(clip_dataset)
    clip_dataloader = DataLoader(clip_dataset, Config.batch_size, sampler=train_sampler, num_workers=4, collate_fn=data_collator) 

    # Create an instance of your model
    model = CustomModel(Config)

    offset = 0
    # Add ckpt here!
    if args.ckpt:
        if dist.get_rank() == 0:
            model.load_state_dict({k.replace('module.', ''): v for k, v in                 
                       torch.load(args.ckpt).items()}, strict=False)
        offset = int(args.ckpt.split("_")[-1].split(".")[0])
        print(f"Loaded ckpt from {args.ckpt} at epoch {offset}")
    
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.module.vision_encoder.parameters()},
            {"params": model.module.caption_encoder.parameters()},
        ], lr=Config.lr
    )
    num_epochs = Config.epochs
    steps_per_epoch = len(clip_dataloader)           # 每个 epoch 的 batch 数
    num_training_steps = steps_per_epoch * Config.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps)
    if offset > 0:
        print(offset)
        scheduler.last_epoch = offset
    
    model.train()
    for epoch in range(offset + 1, num_epochs + 1):
        
        train_sampler.set_epoch(epoch)  
        # cnt = 0
        for batch in tqdm(clip_dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
            video1 = batch["video1"].to(device)
            video2 = batch["video2"].to(device)
            # video1_mask = batch["video1_mask"].to(device)
            # video2_mask = batch["video2_mask"].to(device)
            label = batch["label"].to(device)
            text = batch["caption"]
            pos = batch["pos"]

            logits = model(video1, video2, text, pos, device)
            logits = logits / Config.temperature
            loss = F.cross_entropy(logits, label, reduction="mean") # (bs,cls_num=n) (bs,)

            # Backward loss and optimization every step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step() 

        if dist.get_rank() == 0:
            pred = torch.argmax(logits, dim=1)
            acc = torch.sum(pred == label).item() / logits.size(0)
            logging.warning(f'{loss.item()},{acc}')

            # cnt += 1
        
        if dist.get_rank() == 0 and epoch % 1 == 0 and epoch != num_epochs:
            path = os.path.join(
                save_dir,
                f"clip_epoch_{epoch}.pt"
            )
            save_model(model, path)
       
        print(f"Epoch [{epoch}/{num_epochs}], Batch Loss: {loss.item()}")
    
    if dist.get_rank() == 0:
        path = os.path.join(
                    save_dir,
                    f"clip_epoch_{num_epochs}.pt"
                )
        save_model(model, path)

    print("Training complete.")

def save_model(model, path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, path) # parameters


if __name__ == "__main__":
    train()