from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, CLIPTokenizer, CLIPTextModel
from qwen_vl_utils import process_vision_info
import pickle
import torch
from tqdm import tqdm
import json
import math
import copy
import csv
import os
import logging
import re
from torch.utils.data import DataLoader
from functools import partial
from PIL import Image
from peft import (
    LoraConfig,
    TaskType,
    PeftModel
)
from safetensors.torch import load_file
from finetune_dsp import LazySupervisedDataset
from finetune_e2e import LazySupervisedDataset as LazySupervisedDataset_e2e
from custom_qwen import Qwen2VLForText2ImgConditionalGeneration
from simda_model.pipeline_simda import SimDAPipeline
from simda_model.util import save_videos_grid,save_videos_imgs
from finetune_e2e import DataCollatorForSupervisedDataset
from simda_model.unet import UNet3DConditionModel
# from blackswan import BlackswanDataset
from youcook import YoucookDataset

origin_model_dir = 'dVAR/LLM_Weights/Qwen2-VL-7B-Instruct'
sd_model_dir = "dVAR/LLM_Weights/stable-diffusion-v1-4"

output_dir = 'resources/outputs/'

def collator_e2e(instances, processor):
    samples = [m["samples"] for m in instances]
    messages = [sample['messages'][:-1] for sample in samples]
    exids = [sample["id"] for sample in samples]

    reasons = [m["reason"] for m in instances]
    video_condition = [m["video_condition"] for m in instances]
    

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
   
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to('cuda')

    input_ids_lists = inputs["input_ids"].tolist()
    assert len(messages) == len(input_ids_lists)
 
    batch = dict(
            **inputs,
            exids=exids,
            reasons=reasons,
            video_condition_state=torch.stack(video_condition)
        )

    return batch

def collator(instances, processor):
  
    messages = [m['messages'][:-1] for m in instances]
    exids = [m["id"] for m in instances]

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
   
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to('cuda')

    input_ids_lists = inputs["input_ids"].tolist()
    assert len(messages) == len(input_ids_lists)
 
    batch = dict(
            **inputs,
            exids=exids
        )

    return batch


def run_batch_inference():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        origin_model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(origin_model_dir, max_pixels=64*28*28)

    lora_model_dir = 'ckpts/qwen2vl_7b_one_num/checkpoint-940' # 'ckpts/qwen2vl_7b_youcookii_epoch3_lr1e-5/checkpoint-1468'  
    model = PeftModel.from_pretrained(model, model_id=lora_model_dir) # 测试e2e时要加载2个Lora
    model = model.merge_and_unload()

    output_name = 'qwen2vl_7b_e2e_sele3_random'

    lora2_dir = 'ckpts/qwen2vl_7b_e2e_sele3_random/checkpoint-470' #'ckpts/qwen2vl_7b_e2e_sele3_youcook_lr2e-4_a2/checkpoint-734'
    model = PeftModel.from_pretrained(model, model_id=lora2_dir)
 
    testset = LazySupervisedDataset(split='test', infer = True, K = 3, method='random')
    # testset = YoucookDataset(split='validation', infer = True, K = 3)
   

    dataloader = DataLoader(testset, batch_size=16, shuffle=False,
                            collate_fn=partial(collator, processor=processor))
    
    res_file = os.path.join(output_dir, f"{output_name}.json")
    output_dict = {}  
    with torch.inference_mode():
        for sample in tqdm(dataloader):
            exids = sample.pop('exids')
            
            # Inference     
            generated_ids = model.generate(**sample, max_new_tokens=100)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(sample['input_ids'], generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for exid, output_text in zip(exids, output_texts):
                output_dict[exid] = output_text
                print(exid, output_text)
            
            if len(output_dict) % 500 == 0:
                print(f"====== Saving intermediate results at {len(output_dict)} samples ======")
                with open(res_file, 'w') as file:
                    json.dump(output_dict, file)
   
    
    with open(res_file, 'w') as file:
        json.dump(output_dict, file)
    outfile = make_submit(output_name, data='var')
    print(f"====== Results saved to {outfile} ======")


def run_var_generation():
    # model = Qwen2VLForText2ImgConditionalGeneration.from_pretrained(
    #     origin_model_dir,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     low_cpu_mem_usage=False
    # )

    # default processer
    processor = AutoProcessor.from_pretrained(origin_model_dir, max_pixels=64*28*28)
    
    # lora_model_dir = "ckpts/qwen2vl_2b_l8t16one_num/checkpoint-1878"
    # output_name = 'qwen2vl_2b_l8t16lr2one_num_gen'
    # model = PeftModel.from_pretrained(model, model_id=lora_model_dir)   
    # model = model.merge_and_unload()
    # model.init_unet_only(sd_model_dir, torch.bfloat16)

    unet_dir = "ckpts/qwen2vl_2b_e2e_all_5e-5/checkpoint-234/unet"
    # state_dict = torch.load(os.path.join(unet_dir, 'proj_diff_model.bin'))

    # for k, v in state_dict.items():
    #     if k in model.state_dict():
    #         print(k)
    #         model.state_dict()[k].copy_(v)
    
    # model.to("cuda")

    unet = UNet3DConditionModel.from_pretrained(unet_dir, torch_dtype=torch.bfloat16).to('cuda')
    pipe = SimDAPipeline.from_pretrained(sd_model_dir, unet=unet, torch_dtype=torch.bfloat16).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_slicing()

    testset = LazySupervisedDataset_e2e(split='test')
    testset.list_data = testset.list_data[100:200]
    dataloader = DataLoader(testset, batch_size=4, shuffle=False, collate_fn=partial(collator_e2e, processor=processor))
   
    output_dict = {}  
    with torch.inference_mode():
        for sample in tqdm(dataloader):
            exids = sample.pop('exids')
    
            # Inference text    
            # generated_ids = model.generate(**sample, max_new_tokens=100)  
            # generated_ids_trimmed = [
            #     out_ids[len(in_ids):] for in_ids, out_ids in zip(sample['input_ids'], generated_ids)
            # ]
            # output_texts = processor.batch_decode(
            #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            # )
            # for exid, output_text in zip(exids, output_texts):
            #     output_dict[exid] = output_text
            #     print(exid, output_text)

            output_texts = sample['reasons']
            print(output_texts)
            # Inference video 
            batch_videos = pipe(output_texts, video_condition_state=sample['video_condition_state'], latents=None, video_length=16, height=256, width=256, num_inference_steps=50, guidance_scale=12.5).videos
            for exid, video in zip(exids, batch_videos):
                image_path = os.path.join('/data/processed/VAR_Dataset/arg_imgs/simda_ft2', str(exid))
                save_videos_imgs(video.unsqueeze(0), image_path)
              
    # res_file = os.path.join(output_dir, f"{output_name}.json")
    # with open(res_file, 'w') as file:
    #     json.dump(output_dict, file)
    # print(f"====== Results saved to {res_file} ======")

def make_submit(output_name, data):
    res_file = os.path.join(output_dir, f"{output_name}.json")
    out_file = os.path.join(output_dir, f"{output_name}_submit.json")
    gt_file = "resources/var_test_list_msg0.json"
    if data == 'youcook':
        gt_file = 'resources/youcookii_validation_list_msg.json'

    with open(gt_file) as file:
        gt_contents = json.load(file)
    with open(res_file) as file:
        res_contents = json.load(file)

    submit_dict = {}

    for sample in gt_contents:
        id = sample['id']
        res = res_contents[id]
        if data == 'var':
            match = re.search(r'is that (.*)', res) # VAR
        elif data == 'youcook':
            match = re.search(r'is to (.*)', res) # Youcook2

        if match:
            res = match.group(1)

        submit_dict[id] = [{
            'sentence': res.strip().capitalize(),
            'gt_sentence': sample['messages'][2]['content'][0]['text'].strip().capitalize(), #+'.',
            'is_hypothesis': True,
            'query': sample['messages'][1]['content'][0]['text'],
        }]


    with open(out_file, 'w') as file:
        json.dump(submit_dict, file)
    return out_file


def make_submit_clip(output_name):
    res_file = os.path.join(output_dir, f"{output_name}.json")
    clip_out_file = os.path.join(output_dir, f"{output_name}_clip_submit.json")

    gt_contents = {}
    with open('/newdata2/xw/BlackSwanSuite-Gen/BlackSwanSuite_Gen_Val.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['task'] in ['Forecaster', 'Detective']:
                gt_contents[str(data['q_id'])] = data

    with open(res_file) as file:
        res_contents = json.load(file)

    submit_list1 = [] 
    for id, sample in gt_contents.items():
        res = res_contents[str(id)]
        match = re.search(r'is that (.*)', res)
        if match:
            res = match.group(1)

        submit_list1.append({
            'q_id': id,
            'task': sample['task'],
            'responses': [res.strip().capitalize()] * 3,
            'gt_ref_ans': sample['gt_ref_ans']
        })

    with open(clip_out_file, 'w') as file:
        json.dump(submit_list1, file)

def make_submit_llm(res_file_path):
    output_name = res_file_path.split('/')[-1].split('.')[0]
    llm_out = os.path.join(output_dir, f"{output_name}_llm.json")

    res_contents = json.load(open(res_file_path, 'r'))

    submit_list = [] 

    for id, sample in res_contents.items():
        
        submit_list.append({
            'index': id,
            'task1_responses': [sample[0]['sentence']],
            'task1_gt': [sample[0]['gt_sentence']]
        })     
        

    with open(llm_out, 'w') as file:
        json.dump(submit_list, file)
    
    print(llm_out)


if __name__ == "__main__":
    run_batch_inference()
    # run_var_generation()
    
    # make_submit('qwen2vl_7b_e2e_sele3_youcook_lr2e-4_a2',data='youcook')

    