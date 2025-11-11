from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel
)
import torch
from torch.utils.data import Dataset, ConcatDataset
from util.vision_util import process_vision_info
import json
from dataclasses import dataclass
import pickle
import os
import pathlib
import gc
import copy
import numpy as np
import itertools
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from safetensors.torch import load_file

from custom_qwen_proj import Qwen2VLForText2ImgConditionalGeneration
from einops import rearrange
from util.data_util import _get_seq_frames
from typing import List, Optional, Dict
import csv
import decord
from decord._ffi.base import DECORDError
import random
from util.train_utils import init_logger
from qwen_trainer import QwenProjTrainer
from youcook import YoucookDatasete2e

# os.environ["WANDB_MODE"] = "offline"

training_args = TrainingArguments(
    # deepspeed="./ds_config.json", # deepspeed config file
    output_dir="ckpts/qwen2vl_7b_proj_notemp",
    bf16=True,
    tf32=False,
    seed=42,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    # save_steps=5,
    save_total_limit=10,
    learning_rate=1e-3,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    logging_steps=10,
    gradient_checkpointing=True,
    report_to="tensorboard",  
    ddp_find_unused_parameters=False,
    remove_unused_columns=False # DONOT remove keys which are not in the model's forward
)



class VARDataset(Dataset):

    def __init__(self, split):
        super().__init__()
        self.split = split
        self.meta_data = json.load(open(f'dVAR/var_json/var_{split}_v1.0.json', "r"))  

        self.samples = []
        for key, value in self.meta_data.items():
            for i in range(len(value['events'])):
                if i >= 12: 
                    break
                self.samples.append({
                    'exid': key,
                    'clip_idx': i
                })
                
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
        sample = copy.deepcopy(self.samples[idx])
        exid = sample['exid']
        clip_idx = sample['clip_idx']
        
        with open(f"/newdata1/xw/qwen_data_all/{self.split}/{exid}.pkl", "rb") as f: # <=12 events in one video
            frames = pickle.load(f)
        
        with open(f"dVAR/qwen_data_bound/info_{self.split}/{exid}.pkl", "rb") as f:
            temporal_info = pickle.load(f) # temporal_info[i] is the number of frames in the i-th event
        
        with open(f'dVAR/sd_train_data/var_{self.split}/{exid}-{clip_idx}.pkl', "rb") as f:
            cond_feature = pickle.load(f) # 8-frame condition for diffusion model
        
        gt_frames = frames[clip_idx]
        gt_ids = _get_seq_frames(len(gt_frames), self.max_event_frames)
        gt_frames = [(self.transform(gt_frames[i])*2 - 1) for i in gt_ids]
        
        reason_frames = []
        for i in range(len(frames)):
            if i == clip_idx:
                tmp = frames[i]
                frame_size = tmp[0].size
                frame_mode = tmp[0].mode
                random_frame = Image.fromarray((np.random.rand(frame_size[1], frame_size[0], 3) * 255).astype(np.uint8), mode=frame_mode)
                unknown_video = [random_frame] * len(tmp)
                reason_frames.extend(unknown_video)    
            else:
                reason_frames.extend(frames[i])

        reason_frames = self.add_number(reason_frames, temporal_info)

        # for idx, frame in enumerate(reason_frames):
        #     frame_path = os.path.join('./imgs', f"{idx}.png")
        #     frame.save(frame_path)

        reason =  self.meta_data[exid]['events'][clip_idx]['sentence'].strip()

        sample["messages"] = [
                {
                    "role": "system",
                    "content": "You are an AI assistant able to analyze and infer the most likely explanation for an incomplete set of observation videos."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"The red numbers on each frame represent the event index, with identical numbers for frames in the same event. What most likely happened in event {clip_idx}?  Format your answer to this template: The most likely event is that <The event description.>"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": reason
                        }
                    ]
                }
            ]
        sample['messages'][1]['content'].append(
            {
            "type": "video", 
            "video": reason_frames
            }
        )
 
        return {
            'samples': sample,
            'target_videos': torch.stack(gt_frames),
            'reason': reason,
            'video_condition': cond_feature.detach().cpu()
        }
    

    def add_number(self, frames, temporal_info):
        '''
        Add event index to the frames
        '''
        st = 0  
        new_frames = []
        for event_idx, num_frames in enumerate(temporal_info):
            event_number = str(event_idx)
            for i in range(st,st+num_frames):
                frame = frames[i].copy()
               
                draw = ImageDraw.Draw(frame)
                font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 40)      
                width, height = frame.size
         
                text_bbox = draw.textbbox((0, 0), event_number, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                x = width - text_width
                y = height - text_height - text_height / 3
                
                draw.text((x, y), event_number, font=font, fill='red')
                new_frames.append(frame)

            st += num_frames
       
        return new_frames
    
def find_assistant_content_sublist_indexes(ids_list):
    """
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    """
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(ids_list) - 2):
        # Check if the current and next elements form the start sequence
        if (
            ids_list[i] == 151644
            and ids_list[i + 1] == 77091
            and ids_list[i + 2] == 198
        ):
            start_indexes.append(i + 3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i + 3, len(ids_list) - 1):
                if ids_list[j] == 151645 and ids_list[j + 1] == 198:
                    end_indexes.append(
                        j + 2
                    )  # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

class WebVidDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.meta_file_path = '/newdata2/xw/cby-panda/merged_output.csv'
        self.video_path_list = []

        # for dirpath, dirnames, filenames in os.walk('/newdata2/xw/cby-panda/download'):
        #     for fname in filenames:
        #         full_path = os.path.join(dirpath, fname)
        #         self.video_path_list.append(full_path)
        # with open('/newdata2/xw/cby-panda/file_path.txt', 'w') as f:
        #     for path in self.video_path_list:
        #         f.write(path + '\n')

        with open('/newdata2/xw/cby-panda/good_path2.txt', 'r') as f:
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

        sample = {
            "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": caption
                }
                ]
            },
            {
                "role": "assistant",
                "content": [
                {
                    "type": "text",
                    "text": caption + ''.join([f"[VID{i}]" for i in range(24)])
                }
                ]
            }
            ]
        }

        try:
            vr = decord.VideoReader(video_path, width=256, height=256)
            idx_list = _get_seq_frames(len(vr), self.max_event_frames)
            video = torch.from_numpy(vr.get_batch(idx_list).asnumpy())
            video = rearrange(video, "f h w c -> f c h w")

            return {
                'samples': sample,
                'target_videos': (video / 127.5 - 1.0),  # Normalize to [-1, 1]
                'reason': caption
            }
        except DECORDError as e:
            print(f"Error reading video {video_path}: {e}")
            new_idx = random.randint(0, idx - 1)
            return self.__getitem__(new_idx)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    processor: AutoProcessor

    def __call__(self, instances):
        # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
        # [151644, 77091]
        # (Pdb++) processor.tokenizer.encode("<|im_end|>")
        # [151645]

        samples = [m["samples"] for m in instances]
        target_videos = [m["target_videos"] for m in instances]
        reasons = [m["reason"] for m in instances]

        video_condition = [m["video_condition"] for m in instances]


        messages = [sample['messages'] for sample in samples]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]
        # print(texts[1])
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids_lists = inputs["input_ids"].tolist()
        assert len(messages) == len(input_ids_lists)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)  # -100 is the ignore index in loss function
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0] : begin_end_indexs[1]] = ids_list[
                    begin_end_indexs[0] : begin_end_indexs[1]
                ]
            labels_list.append(label_ids)

        labels_ids = torch.tensor(labels_list, dtype=torch.int64)

        batch = dict(
            **inputs,
            labels=labels_ids,          
            target_videos=torch.stack(target_videos),
            reasons=reasons,
            video_condition_state=torch.stack(video_condition)
        )

        return batch

    

def train():
    """
    2. End-to-end train Qwen2vl and Stable Diffusion for video generation.
    """
    model_dir = 'dVAR/LLM_Weights/Qwen2-VL-7B-Instruct'
    weight_dtype = torch.bfloat16
    model = Qwen2VLForText2ImgConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=weight_dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False
    )

    # Load processor. 
    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_dir, padding_side="right", max_pixels=16*28*28)
    
    model.config.use_cache = False
    
    lora_path = 'ckpts/qwen2vl_7b_one_num/checkpoint-940' #'ckpts/qwen2vl_7b_youcookii_epoch3_lr1e-5/checkpoint-1468' 
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

    sd_model_dir = 'dVAR/LLM_Weights/stable-diffusion-v1-4'
    pretrained_unet_dir = 'ckpts/var_unet_notemp_pretrain/checkpoint-1080' # 'ckpts/youcook_unet_pretrain/checkpoint-734'
    model.init_unet_modules(sd_model_dir, pretrained_unet_dir, weight_dtype)

    for params in model.parameters():
        params.requires_grad = False

    model.enable_input_require_grads()

    for name, params in model.named_parameters():
        if "text_projection" in name:
            params.requires_grad = True 
    
    # for name, params in model.named_parameters():
    #     if params.requires_grad:
    #         print(name)
    
    train_dataset = VARDataset(split='trainval')
    # train_dataset = YoucookDatasete2e(split='training', infer=False)

    trainer = QwenProjTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSupervisedDataset(processor=processor)
    )
    
    model.logger = init_logger(training_args.output_dir)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()  


if __name__ == "__main__": 
    train()

    # dataset1 = VARDataset(split='trainval')
    # a=dataset1.__getitem__(20)
    # print(a['input_ids'])
    # print(a['target_videos'].shape)
    # print(a['target_videos'].min().item(), a['target_videos'].max().item())
    # dataset2 = WebVidDataset()
    # b=dataset2.__getitem__(20)
    # print(b['target_videos'].shape)
    # print(b['target_videos'].min().item(), b['target_videos'].max().item())