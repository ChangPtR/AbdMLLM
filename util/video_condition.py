import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Dict, Union, Optional
import os
import pickle
import json
from tqdm import tqdm

import imageio
import numpy as np


def _get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq

class SemanticAdapter(nn.Module):
    def __init__(self, clip_model_path="dVAR/LLM_Weights/clip-vit-large-patch14"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_path)
        
    def forward(
        self,
        video_frames: List[Image.Image],  # list of PIL Images 
        text: str,
        k: int = 8,
    ) -> torch.Tensor:
        images_inputs = self.processor(images=video_frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.get_image_features(**images_inputs)  # (num_frames, dim)
        frame_features = vision_outputs  # c_v^i

        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**text_inputs)  # (1, dim)
        text_feature = text_outputs[0]  # c_t, shape: (dim,)

        # 3. 归一化以计算余弦相似度
        frame_norms = frame_features / frame_features.norm(dim=-1, keepdim=True)
        text_norm = text_feature / text_feature.norm()

        # 4. 计算相似度 sim(c_v^i, c_t)
        # 结果 shape: (num_frames,)
        sims = torch.matmul(frame_norms, text_norm)

        # 5. 计算权重 gamma^i = softmax(sims)
        gamma = torch.softmax(sims, dim=0)  # shape: (num_frames,)

        # 6. 挑选相似度最高的 k 帧索引
        topk_vals, topk_idx = torch.topk(sims, k=k, largest=True)

        # 7. 构造局部特征 c_local: 拼接所选帧特征
        c_local = frame_features[topk_idx]  # (k, dim)

        c_global = torch.sum(frame_features * gamma.unsqueeze(-1), dim=0)  # (dim,)
        # c_global = c_global.unsqueeze(0)  # (1, dim)
        c_global_expanded = c_global.unsqueeze(0).expand(k, -1)  # (k, dim)
       
        # c_v = torch.cat([c_global, c_local], dim=0)  # (1+k, dim)
        c_v = torch.cat([c_local, c_global_expanded], dim=1)   # (k, 2*dim)
        return c_v
    

class VideoTextDataset(Dataset):
    def __init__(self, split: str = 'trainval'):

        self.video_folder = f'/newdata1/xw/qwen_data_all/{split}'
      
        self.meta_data = json.load(open(f'dVAR/var_json/var_{split}_v1.0.json', "r"))
        self.split = split
           
        self.samples = []
        for key, value in self.meta_data.items():
            for item in value['events']:
                sample = item.copy()
                sample['exid'] = key
                self.samples.append(sample)
                
        print(f'dataset has {len(self.samples)} samples')
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        exid = sample['exid']
        text = sample['sentence']
        clip_idx = sample['clip_idx']
        
        # 加载视频帧
        frames = pickle.load(open(os.path.join(self.video_folder, f'{exid}.pkl'), 'rb'))
        frames_to_select = []
        for i in range(len(frames)):
            if i == clip_idx:
                continue
            frames_to_select.extend(frames[i])
        
        idx_list = _get_seq_frames(len(frames_to_select), 32) # represent oberved video with 32 frames
        frames_to_select = [frames_to_select[i] for i in idx_list]

        return {
            'exid': exid,
            'clip_idx': clip_idx,
            'text': text,
            'frames': frames_to_select
        }

class YoucookVideoTextDataset(Dataset):

    def __init__(self, split='training'):
        super().__init__()
        meta_data = json.load(open('dVAR/youcook_json/youcookii_trainval_v2.0.json', "r"))
        self.list_data = []
        self.split = split
        for key, value in meta_data.items():
            if value['split'] == split:
                value['id'] = key
                self.list_data.append(value)

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        sample = self.list_data[idx]
        exid = sample['id']
        vid = exid.rsplit('-', 1)[0]
        hid = sample['hypothesis']
        
        frames = pickle.load(open(os.path.join('/newdata2/xw/youcook2/frames', f'{vid}-0.pkl'), 'rb'))
        frames_to_select = []
        for i in range(len(frames)):
            if i == hid:
                continue
            frames_to_select.extend(frames[i])
        
        idx_list = _get_seq_frames(len(frames_to_select), 32) # represent oberved video with 32 frames
        frames_to_select = [frames_to_select[i] for i in idx_list]


        return {
            'exid': exid,
            'text': sample['events'][hid]['sentence'],
            'frames': frames_to_select
        }


def get_cv_with_text():

    # output_folder = '/data/processed/VAR_Dataset/qwen_data_bound/test_exp_cv'
    # output_folder = 'dVAR/sd_train_data/trainval'
    # dataset = VideoTextDataset(split='trainval')

    output_folder = 'dVAR/sd_train_data/youcook_training'
    dataset = YoucookVideoTextDataset(split='training')
    # dataset.samples = dataset.samples[:10000]
    model = SemanticAdapter()
    
    # 批量推理
    with torch.no_grad():
        for data in tqdm(dataset):
            exid = data['exid']
            text = data['text']
            frames_list = data['frames']

            if os.path.exists(os.path.join(output_folder, f'{exid}.pkl')):
                continue

            c_vs = model(frames_list, text)
            pickle.dump(c_vs, open(os.path.join(output_folder, f'{exid}.pkl'), 'wb'))


def get_cv_idx_with_text():
    output_folder = 'dVAR/sd_train_data/'
    dataset = VideoTextDataset(split='trainval')
    # dataset.samples = dataset.samples[:10000]
    model = SemanticAdapter()
    
    # 批量推理
    results = {}
    
    with torch.no_grad():
        for data in tqdm(dataset):
            exid = data['exid']
            text = data['text']
            frames_list = data['frames']
            clip_idx = data['clip_idx']
           
            # TODO add continue
            topk_vals, topk_idx = model(frames_list, text)
            # pickle.dump(c_vs, open(os.path.join(output_folder, f'{exid}-{clip_idx}.pkl'), 'wb'))
            key = f'{exid}-{clip_idx}'
            results[key] = {
                'exid': exid,
                'clip_idx': clip_idx, 
                'topk_vals': topk_vals.cpu().numpy().tolist(),
                'topk_idx': topk_idx.cpu().numpy().tolist()
            }
            if len(results) % 500 == 0:
                json.dump(results, open(os.path.join(output_folder, 'var_trainval_ids.json'), 'w'), indent=4)
        
        json.dump(results, open(os.path.join(output_folder, 'var_trainval_ids.json'), 'w'), indent=4)

class TgifDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.gif_folder = 'tgif'
        self.gif_files = []
        for file in os.listdir(self.gif_folder):
            if file.endswith('.gif'):
                self.gif_files.append(os.path.join(self.gif_folder, file))
        
        self.caption_dict = {}
        
        with open('dVAR/TGIF-Release-master/data/tgif-v1.0.tsv', 'r') as f:
            for line in f:
                gif_url, caption = line.strip().split('\t')
                self.caption_dict[gif_url.split('/')[-1]] = caption

        
    def __len__(self):
        return len(self.gif_files)
    
    def __getitem__(self, idx):
        gif_path = self.gif_files[idx]
        gif_id = gif_path.split('/')[-1]

        caption = self.caption_dict[gif_id]
        gif = imageio.get_reader(gif_path)
        frames = []
        
        for frame in gif:
            frame = Image.fromarray(frame).resize((256, 256))
            frames.append(frame)
        
        idx_list = _get_seq_frames(len(frames), 9)
        frames_to_select = [frames[i] for i in idx_list]
            
        return {
            'gifid': gif_id,
            # 'text': caption,
            'frames': frames_to_select
        }

def get_cv():
    output_folder = 'dVAR/sd_train_data/tgif_all'
    os.makedirs(output_folder, exist_ok=True)

    dataset = TgifDataset()
    
    # 初始化CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("dVAR/LLM_Weights/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("dVAR/LLM_Weights/clip-vit-large-patch14")   

    with torch.no_grad():

        for data in tqdm(dataset):
            gifid = data['gifid']
            frames_list = data['frames']
            output_path = os.path.join(output_folder, f'{gifid}.pkl')

            if os.path.exists(output_path):
                continue
                
            images_inputs = processor(images=frames_list, return_tensors="pt").to(device)
            frame_features = model.get_image_features(**images_inputs)  # (num_frames, dim)  
            pickle.dump(frame_features, open(output_path, 'wb'))

if __name__ == "__main__":
    get_cv_with_text()
                
               
                    
      

