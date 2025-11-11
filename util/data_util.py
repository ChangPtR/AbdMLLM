import math
import csv
import json
import copy
from tqdm import tqdm
import torch
import numpy as np
import pickle
from decord import VideoReader, cpu
from PIL import Image
import os
import concurrent.futures
import gc
import itertools
from PIL import Image, ImageDraw, ImageFont

def _get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq

def load_video_1fps(vis_path, duration: float):
    vr = VideoReader(vis_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    
    fps = round(total_frame_num / duration) # save as 1fps for conversion

    frame_idx = [i for i in range(0, total_frame_num, fps)]

    num_frame = len(frame_idx)

    img_array = vr.get_batch(frame_idx).asnumpy()  # (T, H, W, 3)

    img_array = img_array.reshape((1, num_frame, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(num_frame):
        img = Image.fromarray(img_array[0, j])
        width, height = img.size
        short_edge = min(width, height)
        scale = 224 / short_edge
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        
        resized_img = img.resize((new_width, new_height), resample=Image.LANCZOS)
        clip_imgs.append(resized_img)

    return clip_imgs

def process_single_example(exid, sample, split, video_info):
    """多线程处理单个样本的核心函数"""
    output_path = f'dVAR/qwen_data1fps/{split}/{exid}.pkl'
    if os.path.exists(output_path):
        return  

    # 创建线程本地存储的视频帧缓存
    local_frame_cache = {}
    
    video_ids = set(event['video_id'] for event in sample['events'])
    
    for vid in video_ids:
        video_path = f"/data/original/VAR-raw-video/videos/{vid}.mp4"
        local_frame_cache[vid] = load_video_1fps(video_path, video_info[vid][0])
    
    # 处理每个事件并提取帧
    result_frames = []
    for event in sample['events']:
        vid = event['video_id']
        start = math.floor(event['timestamp'][0])
        end = math.ceil(event['timestamp'][1])
        
        # 获取该视频的帧并截取区间
        frames = local_frame_cache[vid]
        end = min(end + 1, len(frames))  # 确保不越界
        result_frames.append(frames[start:end])
    
    # 保存结果到文件
    with open(output_path, 'wb') as f:
        pickle.dump(result_frames, f)
    
    del local_frame_cache
    del result_frames
    gc.collect()

def prepare_frames(split: str):
    examples = json.load(open(f'dVAR/data/var_{split}_v1.0.json', "r"))
   
    video_info={}
    with open('dVAR/data/var_video_duration_v1.0.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:   
            video_info[row[0]]=[float(row[1]), int(row[2])] # duration float, frame_count int   
    
    # 使用线程池并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:  # 根据CPU核心数调整
        futures = []
        for exid, sample in examples.items():
            futures.append(executor.submit(
                process_single_example,
                exid, sample, split, video_info
            ))
        
        # 显示进度条
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass  # 只需等待任务完成
         

def prepare_list_msg(split: str):
    # examples = json.load(open(f'dVAR/var_json/var_{split}_v1.0.json', "r"))
    examples = json.load(open('dVAR/youcook_json/youcookii_trainval_v2.0.json', "r"))
    res = []
    for exid, sample in examples.items():
        if sample['split'] != split:
            continue
        dict ={
            "id": exid,
            "messages": [
            {"role": "system", "content": "You are an AI assistant able to analyze and infer the most likely explanation for an incomplete set of observation videos."},
            {
                "role": "user",
                "content": []
            },
            {
                "role": "assistant",
                "content": []
            }]
        }
        hyid = sample['hypothesis']
        gt = sample["events"][hyid]["sentence"].strip().capitalize()
        dict['hypothesis'] = hyid
        event_num = len(sample['events'])
        if hyid == 0:
            dict['pos'] = 'BEFORE'
        elif hyid == event_num - 1:
            dict['pos'] = 'AFTER'
        else:
            dict['pos'] = 'BETWEEN'
    
        dict['messages'][1]['content'].append({"type": "text", "text": f"The red numbers on each frame represent the event index, with identical numbers for frames in the same event. What is the most likely Action in event {hyid}? Format your answer to this template: The most likely action is to <The Action Description>. **<The Action Description> should begin with a verb (imperative sentence).**"}) 
        
        if split == 'test' or split == 'validation':
            dict['messages'][2]['content'].append({"type": "text", "text": gt})
        elif split == 'trainval':
            gt = gt.strip()
            dict['messages'][2]['content'].append({"type": "text", "text": f"The most likely event is that {gt[0].lower() + gt[1:]}"})
        elif split == 'training':
            gt = gt.strip()
            dict['messages'][2]['content'].append({"type": "text", "text": f"The most likely action is to {gt[0].lower() + gt[1:]}."})  

        res.append(dict)
    
    # json.dump(res, open(f'resources/var_{split}_list_msg0.json', "w"))
    json.dump(res, open(f'resources/youcookii_{split}_list_msg.json', "w"))



def prepare_gt_frames(split: str):
    examples = json.load(open(f'dVAR/data/var_{split}_v1.0.json', "r"))
    res_dict = {}
    frame_dict = {}
    desired_num_frames = 20

    video_info={}
    with open('dVAR/data/var_video_duration_v1.0.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:   
            video_info[row[0]]=[float(row[1]), int(row[2])] # duration float, frame_count int   

    c = 0
    for exid, sample in tqdm(examples.items()):
        if os.path.exists(f'dVAR/qwen_data/{split}/{exid}_hyp.pkl'):  # Check if the file is already processed
            continue
    
        hyid = sample['hypothesis']
        video_name = sample['events'][hyid]['video_id']
        if video_name not in frame_dict:
            video_path = f"/data/original/VAR-raw-video/videos/{video_name}.mp4"
            frame_dict[video_name] = load_video_1fps(video_path, duration = video_info[video_name][0])
        else:
            print('video already loaded!')
     
        tmp = []
        event = sample['events'][hyid]
        st, ed = math.floor(event['timestamp'][0]), math.ceil(event['timestamp'][1])
        ed = min(ed+1, len(frame_dict[video_name]))
        tmp.extend(frame_dict[video_name][st:ed]) # the benifit of 1fps
        
        # sample desired_num_frames from 1fps tmp
        idx_list = _get_seq_frames(len(tmp), desired_num_frames)
        res_dict[exid] = [[tmp[idx] for idx in idx_list]]

        if len(frame_dict) > 100:
            frame_dict = {}  

        c += 1
        if c % 50 == 0:
            print('writing to files...')
            for key in res_dict.keys():
                with open(f'dVAR/qwen_data/{split}/{key}_hyp.pkl', 'wb') as f:
                    pickle.dump(res_dict[key], f)
            res_dict = {}
    
    for key in res_dict.keys():
        with open(f'dVAR/qwen_data/{split}/{key}_hyp.pkl', 'wb') as f:
            pickle.dump(res_dict[key], f)    

def get_bounded_frames(split: str, desired_num_frames=100):

    examples = json.load(open(f'dVAR/data/var_{split}_v1.0.json', "r"))
    for exid, sample in tqdm(examples.items()):
        with open(f"dVAR/qwen_data1fps/{split}/{exid}.pkl", "rb") as f:
            frames_list = pickle.load(f)
        
        hid = sample['hypothesis']
        if hid > 0 and hid < len(frames_list)-1: # between
            list1 = list(itertools.chain(*frames_list[:hid]))
            list2 = list(itertools.chain(*frames_list[hid+1:]))
            total = len(list1) + len(list2)
            num1 = int(len(list1) / total * desired_num_frames)
            num2 = desired_num_frames - num1
            if len(list1) > num1:
                list1 = _uniform_sample(list1, num1)
            if len(list2) > num2:
                list2 = _uniform_sample(list2, num2)
            video_frames = [list1, list2]
        elif hid == 0: # first
            list1 = list(itertools.chain(*frames_list[1:]))
            if len(list1) > desired_num_frames:
                list1 = _uniform_sample(list1, desired_num_frames)
            video_frames = [list1]
        elif hid == len(frames_list)-1: # after
            list1 = list(itertools.chain(*frames_list[:-1]))
            if len(list1) > desired_num_frames:
                list1 = _uniform_sample(list1, desired_num_frames)
            video_frames = [list1]

        with open(f'dVAR/qwen_data_bound/{split}/{exid}.pkl', 'wb') as f:  
            pickle.dump(video_frames, f)
      


def _uniform_sample(frames_list, desired_num_frames):
    idx_list = _get_seq_frames(len(frames_list), desired_num_frames)
    return [frames_list[idx] for idx in idx_list]



if __name__ == "__main__":
    prepare_list_msg("training")    
    # prepare_frames("trainval")
#     prepare_gt_frames("test")
    # get_bounded_frames("test")
    