from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel, AutoConfig
from qwen_vl_utils import process_vision_info
import pickle
import torch
from tqdm import tqdm
import json
import math
import copy
import csv
import os
from collections import defaultdict
from util.data_util import _get_seq_frames
from decord import VideoReader



def model_infer(model, processor, message):

     # Preparation for inference
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message)
    # print(len(video_inputs[0]))

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference     
    generated_ids = model.generate(**inputs, max_new_tokens=200)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text


def qwen_caption_var(split: str):
    origin_model_dir = 'dVAR/LLM_Weights/Qwen2-VL-7B-Instruct'
    # output_dir = f'resources/var_{split}_caption_detail.json'
    output_dir = f'resources/youcookii_{split}_caption_detail.json'

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        origin_model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(origin_model_dir)

    output_dict = defaultdict(dict)  
    if os.path.exists(output_dir):
        with open(output_dir) as file:
            output_dict = json.load(file)
    
    # gt_file = f'resources/var_{split}_list_msg0.json'
    gt_file = 'dVAR/youcook_json/youcookii_annotations_trainval1.json'

    with open(gt_file) as file:
        gt_contents = json.load(file)
    
    model.to('cuda')
    for sample in tqdm(gt_contents[1600:]):
        exid = sample['id']
        if exid in output_dict:
            continue
        # with open(f"dVAR/qwen_data1fps/{split}/{exid}.pkl", "rb") as f:
        #     frames_list = pickle.load(f)

        with open(f'/newdata2/xw/youcook2/frames/{split}/{exid}.pkl', "rb") as f:
            frames_list = pickle.load(f)
        
        output_dict[exid]['hypothesis'] = sample['hypothesis']
        output_dict[exid]['events'] = []

        frames_list = frames_list[:12]
        for frames in frames_list:
            if len(frames) > 50:
                idx_list = _get_seq_frames(len(frames), 50)
                frames = [frames[i] for i in idx_list] 
            message = [
                {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", 
                     "text": 'Describe the core events in terms of WHO is doing WHAT, and in WHERE. Do not include irrelevant visual details. The goal is to provide enough factual information to support reasoning and inference. Return the desciption directly without any prefix.'}
                    
                ]
                }
            ]
            output_text = model_infer(model, processor, message)
       
            output_dict[exid]['events'].append(output_text)

        assert len(output_dict[exid]['events']) == len(frames_list)
        print(exid, len(frames_list),  output_dict[exid]['events'])
        
        if len(output_dict) % 50 == 0:
            with open(output_dir, 'w') as file:
                json.dump(output_dict, file)
            print(f"====== {len(output_dict)} Results saved to {output_dir} ======")

    with open(output_dir, 'w') as file:
        json.dump(output_dict, file)
    print(f"====== Results saved to {output_dir} ======")


def qwen_caption_cookii(split: str):
    origin_model_dir = '/data/xw/VAR/LLM_Weights/Qwen2-VL-7B-Instruct'
    output_dir = f'resources/youcookii_{split}_caption_detail.json'

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        origin_model_dir,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(origin_model_dir)

    output_dict = defaultdict(dict)  
    if os.path.exists(output_dir):
        with open(output_dir) as file:
            output_dict = json.load(file)
    
    gt_file = '/data/xw/VAR/youcook_json/youcookii_annotations_trainval2.json'

    with open(gt_file) as file:
        gt_contents = json.load(file)['database']
    
    model.to('cuda')
    for exid, sample in tqdm(gt_contents.items()):
        if exid in output_dict:
            print(f"Skip {exid}, already processed.")
            continue
        if sample['subset'] != split:
            print(f"Skip {exid}, not in {split} subset.")
            continue
        
        output_dict[exid]={'events': []}
        
        with open(f'/newdata2/xw/youcook2/frames/{exid}-0.pkl', "rb") as f:
            frames_list = pickle.load(f)

        frames_list = frames_list[:12]
        for frames in frames_list:
            if len(frames) > 50:
                idx_list = _get_seq_frames(len(frames), 50)
                frames = [frames[i] for i in idx_list] 
            message = [
                {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", 
                     "text": 'Describe the core events in terms of WHO is doing WHAT, and in WHERE. Do not include irrelevant visual details. The goal is to provide enough factual information to support reasoning and inference. Return the desciption directly without any prefix.'}
                    
                ]
                }
            ]
            output_text = model_infer(model, processor, message)
       
            output_dict[exid]['events'].append(output_text)

        assert len(output_dict[exid]['events']) == len(frames_list)
        print(exid, len(frames_list),  output_dict[exid]['events'])
        
        if len(output_dict) % 50 == 0:
            with open(output_dir, 'w') as file:
                json.dump(output_dict, file)
            print(f"====== {len(output_dict)} Results saved to {output_dir} ======")

    with open(output_dir, 'w') as file:
        json.dump(output_dict, file)
    print(f"====== Results saved to {output_dir} ======")

def qwen_caption_black():
    origin_model_dir = 'dVAR/LLM_Weights/Qwen2-VL-7B-Instruct'
    output_dir = f'resources/blackswan_caption_detail1600.json'

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        origin_model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # default processer
    processor = AutoProcessor.from_pretrained(origin_model_dir)

    output_dict = defaultdict(dict)  
    if os.path.exists(output_dir):
        with open(output_dir) as file:
            output_dict = json.load(file)
    
    gt_contents = []
    with open('/newdata2/xw/BlackSwanSuite-Gen/BlackSwanSuite_Gen_Val.jsonl', 'r') as file:
        for line in tqdm(file):
            data = json.loads(line)
            if data['task'] in ['Forecaster', 'Detective']:
                gt_contents.append(data)
    
    model.to('cuda')
    for sample in tqdm(gt_contents[800:]):
        q_id = sample['q_id']
        if q_id in output_dict:
            continue
        
        output_dict[q_id]['events'] = []
        paths = []
        for key in ['preevent_file_name','postevent_file_name']:
            if sample[key] != "":
                paths.append(('/newdata2/xw/BlackSwanSuite-Gen/' + sample[key]))
        
        for vid_path in paths:
            
            message = [
                {
                "role": "user",
                "content": [
                    {"type": "video", "video": vid_path},
                    {"type": "text", 
                     "text": 'Describe the core events in terms of WHO is doing WHAT, and in WHERE. Do not include irrelevant visual details. The goal is to provide enough factual information to support reasoning and inference. Return the desciption directly without any prefix.'}
                    
                ]
                }
            ]
            output_text = model_infer(model, processor, message)
       
            output_dict[q_id]['events'].append(output_text)

        print(q_id, output_dict[q_id]['events'])
        
        if len(output_dict) % 100 == 0:
            with open(output_dir, 'w') as file:
                json.dump(output_dict, file)
            print(f"====== {len(output_dict)} Results saved to {output_dir} ======")

    with open(output_dir, 'w') as file:
        json.dump(output_dict, file)
    print(f"====== Results saved to {output_dir} ======")
   

if __name__ == "__main__":
    # qwen_caption_black()
    # clean_res()
    qwen_caption_cookii('training')
