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
from torch.utils.data import Dataset
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

from custom_qwen import Qwen2VLForText2ImgConditionalGeneration
from qwen_trainer import QwenTrainer
from einops import rearrange
from util.data_util import _get_seq_frames
from util.train_utils import init_logger
from youcook import YoucookDatasete2e

# os.environ["WANDB_MODE"] = "offline"

training_args = TrainingArguments(
    # deepspeed="./ds_config.json", # deepspeed config file
    output_dir="ckpts/qwen2vl_7b_e2e_sele3_random",
    bf16=True,
    tf32=False,
    seed=42,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    evaluation_strategy="no",
    save_strategy="epoch",
    # save_strategy="steps",
    # save_steps=5,
    save_total_limit=5,
    learning_rate=1e-4,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    logging_steps=5,
    gradient_checkpointing=True,
    report_to="tensorboard",  
    ddp_find_unused_parameters=False,
    remove_unused_columns=False # DONOT remove keys which are not in the model's forward
)


class LazySupervisedDataset(Dataset):

    def __init__(self, split, know = False, infer = True, K = 3, clip=False):

        super().__init__()
        self.list_data = json.load(open(f'resources/var_{split}_list_msg0.json', "r")) 
        self.list_data = self.list_data
        self.know_data = json.load(open(f'resources/var_{split}_knowledge.json', "r"))  
        self.infer_data = json.load(open(f'resources/var_{split}_infers_detail.json', "r")) 
        self.selection_data = pickle.load(open(f'resources/selections/scores_{split}_detail.pkl', "rb"))
        self.K = K
        self.split = split
        self.know = know
        self.infer = infer
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.CenterCrop(256),  
            transforms.ToTensor(),  
        ])
        self.max_event_frames = 8

        if self.method == 'clip':
            self.clip_selection = json.load(open(f'resources/var_{split}_infers_clip_top3.json', "r"))
        

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.list_data[idx])
        exid = sample['id']

        if self.know or self.infer:
            query= sample['messages'][1]['content'][0]['text']
            query += ' You may refer to the following information if necessary.'

            if self.know:
                query += '**Here is some related commonsense knowledge: '+ self.know_data[exid]

            if self.infer:
                scores_arr = self.selection_data[exid]
                tmp = ''
                if self.K < 10:       
                    if self.method == 'clip':
                        res = self.clip_selection[exid]["top3_sentences"]
                        for i in range(len(res)):
                            tmp += f' {i+1}-{res[i]}'
                    elif self.method == 'random':
                        select_ids = np.random.choice(len(scores_arr), self.K, replace=False)  
                        for i in range(1, self.K + 1):
                            tmp += f' {i}-{self.infer_data[exid]["infers"][select_ids[i - 1]]}'
                    else:        
                        select_ids = np.argsort(-scores_arr)[:self.K] # top K selection ids  
                        for i in range(1, self.K+1):
                            tmp += f' {i}-{self.infer_data[exid]["infers"][select_ids[i-1]]}'
                   

                else:
                    for str in self.infer_data[exid]["infers"]:
                        tmp += f' {str}'
                query += ' **Here are some top-ranking inferences:'+ tmp

            sample['messages'][1]['content'][0]['text'] = query
        
        with open(f"dVAR/qwen_data_bound/{self.split}/{exid}.pkl", "rb") as f: # <=12 events in one video
            frames = pickle.load(f)
        
        with open(f"dVAR/qwen_data_bound/info_{self.split}/{exid}.pkl", "rb") as f:
            temporal_info = pickle.load(f) # temporal_info[i] is the number of frames in the i-th event
        
        with open(f"dVAR/qwen_data_bound/gt_{self.split}/{exid}.pkl", "rb") as f: # 16-frame gt video to train diffusion model
            gt_frames = pickle.load(f)
        
        with open(f'dVAR/qwen_data_bound/{self.split}_exp_cv/{exid}.pkl', "rb") as f:
            cond_feature = pickle.load(f) # 8-frame condition for diffusion model
        
        gt_ids = _get_seq_frames(len(gt_frames), self.max_event_frames)
        gt_frames = [(self.transform(gt_frames[i])*2 - 1) for i in gt_ids]
        # gt_frames = [gt_frames[i] for i in gt_ids]

        frames = self.add_number(frames, temporal_info)

        # for idx, frame in enumerate(gt_frames):
        #     frame_path = os.path.join('./test_img', f"{idx}.png")
        #     frame.save(frame_path)
           
        # print(gt_frames[0].shape)

        sample['messages'][1]['content'].append(
            {
            "type": "video", 
            "video": frames
            }
        )
        reason = self.list_data[idx]['messages'][2]['content'][0]['text'].replace('The most likely event is that', '').strip().capitalize()
        # print(sample)
        # print(reason)
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
    
    lora_path = 'ckpts/qwen2vl_7b_one_num/checkpoint-940'  #  'ckpts/qwen2vl_7b_youcookii_epoch3_lr1e-5/checkpoint-1468'
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

    sd_model_dir = 'dVAR/LLM_Weights/stable-diffusion-v1-4'
    pretrained_unet_dir = 'dVAR/LLM_Weights/unet_pretrain' # 'ckpts/youcook_unet_pretrain/checkpoint-734'
    model.init_unet_modules(sd_model_dir, pretrained_unet_dir)
 
    projector_state_dict = torch.load('ckpts/qwen2vl_7b_one_num_proj_e-3_2epoch/checkpoint-234/projector_model.bin') # 'ckpts/qwen2vl_7b_proj_youcook1e-5_ep2/checkpoint-367/projector_model.bin'
    model.load_state_dict(projector_state_dict,strict=False)

    ex_modules = ["text_projection"]
    for name, module in model.named_modules():
        if 'clip_text_encoder' in name and 'self_attn' in name:
            ex_modules.append(name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LORA_R
        lora_alpha=32,  # LORA_ALPHA
        lora_dropout=0.05,  # LORA_DROPOUT
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        exclude_modules=ex_modules
    )
    model = get_peft_model(model, lora_config)

    model.enable_input_require_grads()

    for name, params in model.named_parameters():
        if any(key in name for key in ["temp_adapter","adapter_s","adapter_ffn","text_projection"]):
            params.requires_grad = True 
    
    # for name, params in model.named_parameters():
    #     if params.requires_grad:
    #         print(name)
    # return

    trainer = QwenTrainer(
        model=model,
        args=training_args,
        train_dataset=LazySupervisedDataset(split='trainval'), #  YoucookDatasete2e(split='training', infer=True, K=3)
        data_collator=DataCollatorForSupervisedDataset(processor=processor)
    )

    model.base_model.model.logger = init_logger(training_args.output_dir)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()  


if __name__ == "__main__": 
    train()
    # dataset = LazySupervisedDataset(split='trainval')
    # dataset.__getitem__(0)