from transformers import (
    TrainingArguments,
    Trainer
)

import torch
from torch.utils.data import Dataset, ConcatDataset
import json
from dataclasses import dataclass
import pickle
import os
import pathlib
import gc
import copy
import numpy as np
import itertools
from PIL import Image, ImageDraw, ImageFont,ImageSequence
from torchvision import transforms
from safetensors.torch import load_file

from einops import rearrange
from diffusion_model.sd_model import SDModel
from util.data_util import _get_seq_frames
import imageio
from torch.distributed.elastic.multiprocessing.errors import record
import csv
import decord
from decord._ffi.base import DECORDError
import random
from transformers import CLIPProcessor, CLIPModel

training_args = TrainingArguments(
    output_dir="ckpts/var_unet_notemp_pretrain",
    bf16=True,
    tf32=False,
    seed=42,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    eval_strategy="no",
    save_strategy="epoch",
    # save_strategy="steps",
    # save_steps=5,
    save_total_limit=5,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type='cosine',
    logging_steps=10,
    report_to="none",  
    remove_unused_columns=False # DONOT remove keys which are not in the model's forward
)


class VARDataset(Dataset):

    def __init__(self, split):
        super().__init__()
        self.split = split
        self.meta_data = json.load(open(f'dVAR/var_json/var_{split}_v1.0.json', "r"))    
        self.samples = []
        for key, value in self.meta_data.items():
            for i, item in enumerate(value['events']):
                if i >= 12: 
                    break
                sample = item.copy()
                sample['exid'] = key
                sample['clip_idx'] = i
                self.samples.append(sample)
                
        print(f'vardataset has {len(self.samples)} samples')

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.CenterCrop(256),  
            transforms.ToTensor(),  
        ])
        self.max_event_frames = 8
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        exid = sample['exid']
        clip_idx = sample['clip_idx']

        with open(f'dVAR/sd_train_data/var_{self.split}/{exid}-{clip_idx}.pkl', "rb") as f:
            cond_feature = pickle.load(f) 

        with open(f"/newdata1/xw/qwen_data_all/{self.split}/{exid}.pkl", "rb") as f:
            frames = pickle.load(f)
       
        gt_frames = frames[clip_idx]
        gt_ids = _get_seq_frames(len(gt_frames), self.max_event_frames)
        gt_frames = [(self.transform(gt_frames[i])*2 - 1) for i in gt_ids]

        return {
            'target_videos': gt_frames,
            'reason': sample['sentence'],
            'video_condition': cond_feature.detach().cpu()
        }

class YoucookDataset(Dataset):

    def __init__(self, split):
        super().__init__()
        self.split = split
        meta_data = json.load(open('dVAR/youcook_json/youcookii_trainval_v2.0.json', "r"))
        self.samples = []
        for key, value in meta_data.items():
            if value['split'] == split:
                value['exid'] = key
                self.samples.append(value)
                
        print(f'Youcook has {len(self.samples)} samples')

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.CenterCrop(256),  
            transforms.ToTensor(),  
        ])
        self.max_event_frames = 8
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        exid = sample['exid']
        hid = sample['hypothesis']
        vid = exid.rsplit('-', 1)[0]

        with open(f'dVAR/sd_train_data/youcook_{self.split}/{exid}.pkl', "rb") as f:
            cond_feature = pickle.load(f) 

        frames = pickle.load(open(os.path.join('/newdata2/xw/youcook2/frames', f'{vid}-0.pkl'), 'rb'))
       
        gt_frames = frames[hid]
        gt_ids = _get_seq_frames(len(gt_frames), self.max_event_frames)
        gt_frames = [(self.transform(gt_frames[i])*2 - 1) for i in gt_ids]

        return {
            'target_videos': gt_frames,
            'reason': sample['events'][hid]['sentence'],
            'video_condition': cond_feature.detach().cpu()
        }

class WebVidDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.meta_file_path = '/newdata2/xw/webvid-10M/data/train/partitions/0000.csv'
        self.video_path_list = []

        with open('/newdata2/xw/cby-panda/file_path.txt', 'r') as f:
            for line in f:
                self.video_path_list.append(line.strip())
        
        print(f'webvid dataset has {len(self.video_path_list)} samples')
        
        self.meta_data = {}
        with open(self.meta_file_path, mode='r', encoding='utf-8') as file:
            rows = list(csv.reader(file))
            for row in rows[1:]:  # Skip header row
                self.meta_data[row[0]] = row[4]

        self.max_event_frames = 16

    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, idx): 
        video_path = self.video_path_list[idx]
        video_name = os.path.basename(video_path)[:-4]
        caption = self.meta_data[video_name].replace("\n", "").strip() 
        # print(caption)

        try:
            vr = decord.VideoReader(video_path, width=256, height=256)
            idx_list = _get_seq_frames(len(vr), self.max_event_frames)
            video = torch.from_numpy(vr.get_batch(idx_list).asnumpy())
            video = rearrange(video, "f h w c -> f c h w")

            cond_list = _get_seq_frames(len(vr), 8)
            condition_frames = torch.from_numpy(vr.get_batch(cond_list).asnumpy())
            condition_frames = rearrange(condition_frames, "f h w c -> f c h w")

            return {
                'target_videos': (video / 127.5 - 1.0),  # Normalize to [-1, 1]
                'reason': caption,
                'condition_frames': condition_frames
            }
        except DECORDError as e:
            print(f"Error reading video {video_path}: {e}")
            new_idx = random.randint(0, idx - 1)
            return self.__getitem__(new_idx)

       
@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances):
        # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
        # [151644, 77091]
        # (Pdb++) processor.tokenizer.encode("<|im_end|>")
        # [151645]

        target_videos = [torch.stack(m["target_videos"]) for m in instances]
        reasons = [m["reason"] for m in instances]
        video_conditions = [m["video_condition"] for m in instances]
  
        batch = dict(
            target_videos=torch.stack(target_videos),
            reasons=reasons,
            video_condition_state=torch.stack(video_conditions)
        )

        return batch
    
@record
def train():

    weight_dtype = torch.bfloat16
    unet_model_dir = 'dVAR/LLM_Weights/stable-diffusion-v1-4'

    model = SDModel(unet_model_dir, weight_dtype)
    for param in model.parameters():
        param.requires_grad = False

    for name, params in model.named_parameters():
        if any(key in name for key in ["temp_adapter","adapter_s","adapter_ffn"]):
            params.requires_grad = True 
    

    train_dataset = VARDataset(split='trainval')
    # train_dataset = YoucookDataset(split='training')

    trainer = SDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSupervisedDataset()
    )
    

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()  
    
class SDTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call = False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        try:
            self.model.unet.save_pretrained(output_dir)
        except FileNotFoundError:
            pass
    
        self.save_state()

if __name__ == "__main__": 
    train()
    # dataset = TGIFDataset()[1212]
    # print(dataset['reason'])
    # print(dataset['video_condition'].shape)
    # for d in dataset:
    #     pass