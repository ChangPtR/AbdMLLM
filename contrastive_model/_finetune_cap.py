from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
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

from util.data_util import _get_seq_frames


training_args = TrainingArguments(
    deepspeed="./ds_config.json", # deepspeed config file
    output_dir="ckpts/caption/qwen2vl_7b",
    bf16=True,
    tf32=True,
    seed=42,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=800,
    save_total_limit=10,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    logging_steps=50,
    gradient_checkpointing=True,
    remove_unused_columns=False # DONOT remove keys which are not in the model's forward
)

lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # LORA_R
        lora_alpha=16,  # LORA_ALPHA
        lora_dropout=0.05,  # LORA_DROPOUT
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )

class LazySupervisedDataset(Dataset):

    def __init__(self, split):
        super().__init__()
        self.split = split
        self.dict_data = json.load(open(f'dVAR/data/var_{split}_v1.0.json', "r"))
        self.list_data = self._dict2list()

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        '''
            'exid': k,
            'clip_idx': e['clip_idx'],
            'clip_tot': e['clip_tot'],
            'sentence': e['sentence'] 
        '''
        sample = self.list_data[idx]
        exid = sample['exid']

        with open(f"dVAR/qwen_data1fps/{self.split}/{exid}.pkl", "rb") as f:
            frames_list = pickle.load(f)

        assert len(frames_list) == sample['clip_tot']

        frames = frames_list[sample['clip_idx']]   
        if len(frames) > 32:
            idx_list = _get_seq_frames(len(frames), 32)
            frames = [frames[i] for i in idx_list] 

        conversations = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": "Provide a one-sentence detailed caption for the provided video."},
                    ],
                },
                {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample['sentence']
                    }
                ]
                }
            ]
        return conversations
    
    def _dict2list(self):
        list_data = []
        for k, v in self.dict_data.items():
            for e in v['events']:
                list_data.append({
                    'exid': k,
                    'clip_idx': e['clip_idx'],
                    'clip_tot': e['clip_tot'],
                    'sentence': e['sentence'].strip() 
                })
        return list_data

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

        messages = [m for m in instances]
        
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
        )

        return batch

    

def train():
    model_dir = '/data/LLM_Weights/Qwen/Qwen2-VL/Qwen2-VL-7B-Instruct'
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
   
    processor = AutoProcessor.from_pretrained(model_dir, padding_side="right", max_pixels=16*28*28)

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=LazySupervisedDataset(split='trainval'),
        data_collator=DataCollatorForSupervisedDataset(processor=processor)
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()  

if __name__ == "__main__": 
    train()
    # dataset = LazySupervisedDataset(split='trainval')
    # print(dataset[20])