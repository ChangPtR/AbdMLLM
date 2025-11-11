import json
from collections import defaultdict
import json
import random
import os
from tqdm import tqdm
from PIL import Image
import pickle
import csv
import concurrent.futures
import gc
import math
from util.data_util import load_video_1fps
from torch.utils.data import Dataset
import numpy as np
from util.data_util import _get_seq_frames
import copy
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms

class CausalYoucookDataset(Dataset):
    def __init__(self, split, mode):
        '''
        mode=trainval: positive_texts, negative_texts
        mode=predict: possible infers
        '''
        self.mode = mode
        self.split = split
        self.video_feature_dir = '/newdata2/xw/youcook2/features'
  
        self.frame_to_second = self._load_duration() # 每个特征=多少秒
        self.max_v_len = 300

        self.positive_texts = json.load(open('dVAR/youcook_json/youcookii_trainval_v2.0.json', "r"))
        if mode == 'trainval':       
            if split == 'training':
                negative_texts = json.load(open(f'resources/youcookii_{split}_negative.json', "r"))    
            elif split == 'validation':
                negative_texts = json.load(open(f'resources/youcookii_{split}_infers_detail.json', "r")) 
        elif mode == 'predict':
            negative_texts = json.load(open(f'resources/youcookii_{split}_infers_detail.json', "r"))          
        else:  
            raise NotImplementedError()   
        
        self.negative_texts = list(negative_texts.items())           
    
    def __len__(self):
        return len(self.negative_texts)
    
    def __getitem__(self, idx):
        exid = self.negative_texts[idx][0]
        neg_dict = self.negative_texts[idx][1]
        if self.mode == 'trainval': # only train mode has positive sample
            hid = self.positive_texts[exid]["hypothesis"]
            pos_text = self.positive_texts[exid]["events"][hid]["sentence"]+'.'
            if self.split == 'training':
                all_texts = neg_dict['neg']
            elif self.split == 'validation':
                all_texts = neg_dict['infers']
            idx = np.random.randint(0, len(all_texts), dtype=np.int64)
            all_texts.insert(idx, pos_text) # insert positive text for eval
            # print(all_texts)
            # print(idx, pos_text)
        elif self.mode == 'predict':
            all_texts = neg_dict['infers']
            idx = -1

        pos = neg_dict['pos']

        video1, video2 = self.convert_example_to_features(exid) # numpy arrays features for frames

        # video1, video1_mask = self.pad_video(video1, self.max_v_len)
        # video2, video2_mask = self.pad_video(video2, self.max_v_len)
        # print(len(video1), len(video2))
        video1 = self.pad_video(video1, self.max_v_len)
        video2 = self.pad_video(video2, self.max_v_len)
        
        return {
            'exid': exid,
            'video1': video1.astype(np.float32), # (mvlen, 3072)
            'video2': video2.astype(np.float32), # (mvlen, 3072)
            # 'video1_mask': video1_mask.astype(np.bool),
            # 'video2_mask': video2_mask.astype(np.bool),
            'label': idx,
            'caption': all_texts,
            'pos': pos
        }

    def _load_duration(self):
        """https://github.com/salesforce/densecap/blob/master/data/anet_dataset.py#L120
        Since the features are extracted not at the exact 0.5 secs. To get the real time for each feature,
        use `(idx + 1) * frame_to_second[vid_name] `
        """
        frame_to_second = {}
        sampling_sec = 4  # hard coded, only support 0.5
        with open('dVAR/youcook_json/validation_duration_totalframe.csv', "r") as f:
            next(f)  # Skip the header row
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_to_second[vid_name] = float(vid_dur)*math.ceil(float(vid_frame)*1./float(vid_dur)*sampling_sec)*1./float(vid_frame)
        
        with open('dVAR/youcook_json/training_duration_totalframe.csv', "r") as f:
            next(f)  # Skip the header row
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_to_second[vid_name] = float(vid_dur)*math.ceil(float(vid_frame)*1./float(vid_dur)*sampling_sec)*1./float(vid_frame)
        return frame_to_second
    
    def convert_example_to_features(self, exid):
        sample = self.positive_texts[exid] # get video info
        video_names = set()
        for e in sample['events']:
            if e['clip_idx'] != sample['hypothesis']:
                video_names.add(e['video_id'])
   
        video_features = {} # video_features to make a single example
        for video_name in video_names: 
            feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name)) # nfrm, 2048
            feat_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name)) # nfrm, 1024
            video_feature = np.concatenate([np.load(feat_path_resnet), np.load(feat_path_bn)], axis=1) # nfrm, 3072
            video_features[video_name] = video_feature

        tmp1, tmp2 =[], [] # single example features without hypothesis
        num_sen = len(sample["events"])
        for clip_idx in range(num_sen):
            if clip_idx < sample["hypothesis"]: 
                tmp1.extend(self.clip_event_feature(sample["events"][clip_idx], video_features))
            elif clip_idx > sample["hypothesis"]:
                tmp2.extend(self.clip_event_feature(sample["events"][clip_idx], video_features))
            
        if len(tmp1) and len(tmp2): # BETWEEN
            return np.array(tmp1), np.array(tmp2)
        else: # AFTER/BEFORE
            single_video_features = tmp1 if len(tmp1) else tmp2
            num = len(single_video_features)
            res1 = single_video_features[:int(num/2)]
            res2 = single_video_features[int(num/2):]
            return np.array(res1), np.array(res2)
        
    
    def clip_event_feature(self, event, video_features):
        video_name = event['video_id']
        timestamp = event['timestamp']
        frm2sec = self.frame_to_second[video_name]
        raw_feat = video_features[video_name]
        st, ed = self._convert_to_feat_index_st_ed(len(raw_feat), timestamp, frm2sec)
        return raw_feat[st:ed + 1]
    
    def _convert_to_feat_index_st_ed(self, feat_len, timestamp, frm2sec):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(round(timestamp[1] / frm2sec))
        ed = min(ed, feat_len-1)
        st = min(st, ed-1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed
    
    def pad_video(self, video, max_v_len):

        new_video = video
        idx_list = _get_seq_frames(len(video), max_v_len)
        new_video = video[idx_list]
        return new_video

class YoucookDataset(Dataset):

    def __init__(self, split, infer = False, K = 3):
        super().__init__()
        self.list_data = json.load(open(f'resources/youcookii_{split}_list_msg.json', "r"))   
        self.infer_data = json.load(open(f'resources/youcookii_{split}_infers_detail.json', "r")) 
        self.selection_data = pickle.load(open(f'resources/selections/scores_youcookii_{split}_detail.pkl', "rb"))
        self.split = split

        self.infer = infer
        self.K = K
        self.max_gen_frames = 8
        self.max_reason_frames = 16

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.list_data[idx])
        exid = sample['id']
        vid = exid.rsplit('-', 1)[0]
        hid = sample['hypothesis']

        if self.infer:
            query= sample['messages'][1]['content'][0]['text']
            query += ' You may refer to the following information if necessary.'
         
            scores_arr = self.selection_data[exid]
            tmp = ''
            if self.K < 10:               
                select_ids = np.argsort(-scores_arr)[:self.K] # top K selection ids  
                for i in range(1, self.K+1):
                    tmp += f' {i}-{self.infer_data[exid]["infers"][select_ids[i-1]]}'
            else:
                for str in self.infer_data[exid]["infers"]:
                    tmp += f' {str}'
            query += ' **Here are some top-ranking inferences:'+ tmp

            sample['messages'][1]['content'][0]['text'] = query
        
        with open(f"/newdata2/xw/youcook2/frames/{vid}-0.pkl", "rb") as f: # <=12 events in one video
            frames = pickle.load(f)
        frames = frames[:12]

        frame_list = []
        for i in range(len(frames)):
            if i == hid:
                frame_size = frames[i][0].size
                frame_mode = frames[i][0].mode
                frm_num = min(len(frames[i]),self.max_reason_frames)
                random_frame = Image.fromarray((np.random.rand(frame_size[1], frame_size[0], 3) * 255).astype(np.uint8), mode=frame_mode)
                unknown_video = [random_frame] * frm_num
                add_number(unknown_video, i)
                frame_list.extend(unknown_video)
            
            else:
                event = frames[i]
                if len(event) > self.max_reason_frames:
                    idx_list = _get_seq_frames(len(event), self.max_reason_frames)
                    event = [event[idx] for idx in idx_list]
                add_number(event, i)
                frame_list.extend(event)
       
        # for idx, frame in enumerate(frame_list):
        #     frame_path = os.path.join('./imgs', f"{idx}.png")
        #     frame.save(frame_path)

        sample['messages'][1]['content'].append(
            {
            "type": "video", 
            "video": frame_list
            }
        )
        # print(sample)
        return sample


class YoucookDatasete2e(Dataset):

    def __init__(self, split, infer, K = 3):
        super().__init__()
        self.list_data = json.load(open(f'resources/youcookii_{split}_list_msg.json', "r"))   
        self.split = split
        self.infer = infer
        self.K = K
        self.infer_data = json.load(open(f'resources/youcookii_{split}_infers_detail.json', "r")) 
        self.selection_data = pickle.load(open(f'resources/selections/scores_youcookii_{split}_detail.pkl', "rb"))

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.CenterCrop(256),  
            transforms.ToTensor(),  
        ])
        self.max_gen_frames = 8
        self.max_reason_frames = 16

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.list_data[idx])
        exid = sample['id']
        vid = exid.rsplit('-', 1)[0]
        hid = sample['hypothesis']

        if self.infer:
            query= sample['messages'][1]['content'][0]['text']
            query += ' You may refer to the following information if necessary.'
         
            scores_arr = self.selection_data[exid]
            tmp = ''
            if self.K < 10:               
                select_ids = np.argsort(-scores_arr)[:self.K] # top K selection ids  
                for i in range(1, self.K+1):
                    tmp += f' {i}-{self.infer_data[exid]["infers"][select_ids[i-1]]}'
            else:
                for str in self.infer_data[exid]["infers"]:
                    tmp += f' {str}'
            query += ' **Here are some top-ranking inferences:'+ tmp

            sample['messages'][1]['content'][0]['text'] = query
        
        with open(f"/newdata2/xw/youcook2/frames/{vid}-0.pkl", "rb") as f: # <=12 events in one video
            frames = pickle.load(f)

        frame_list = []

        for i in range(len(frames)):
            if i == hid:
                frame_size = frames[i][0].size
                frame_mode = frames[i][0].mode
                frm_num = min(len(frames[i]),self.max_reason_frames)
                random_frame = Image.fromarray((np.random.rand(frame_size[1], frame_size[0], 3) * 255).astype(np.uint8), mode=frame_mode)
                unknown_video = [random_frame] * frm_num
                add_number(unknown_video, i)
                frame_list.extend(unknown_video)
            
            else:
                event = frames[i]
                if len(event) > self.max_reason_frames:
                    idx_list = _get_seq_frames(len(event), self.max_reason_frames)
                    event = [event[idx] for idx in idx_list]
                add_number(event, i)
                frame_list.extend(event)
       
        # for idx, frame in enumerate(frame_list):
        #     frame_path = os.path.join('./imgs', f"{idx}.png")
        #     frame.save(frame_path)

        sample['messages'][1]['content'].append(
            {
            "type": "video", 
            "video": frame_list
            }
        )
        # print(sample)
        reason = sample['messages'][2]['content'][0]['text'].replace('The most likely action is to','').strip().capitalize()
        # print(reason)

        with open(f'dVAR/sd_train_data/youcook_{self.split}/{exid}.pkl', "rb") as f:
            cond_feature = pickle.load(f) 
        
        gt_frames = frames[hid]
        gt_ids = _get_seq_frames(len(gt_frames), self.max_gen_frames)
        gt_frames = [(self.transform(gt_frames[i])*2 - 1) for i in gt_ids]

        return {
            'samples': sample,
            'target_videos': torch.stack(gt_frames),
            'reason': reason,
            'video_condition': cond_feature.detach().cpu()
        }


def add_number(frames, event_number):
    event_number = str(event_number)
    for frame in frames:
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 40)      
        width, height = frame.size
    
        text_bbox = draw.textbbox((0, 0), event_number, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = width - text_width
        y = height - text_height - text_height / 3
        
        draw.text((x, y), event_number, font=font, fill='red')


def video_val2train():

    # 读取JSON数据
    with open('dVAR/youcook_json/youcookii_annotations_trainval1.json', 'r') as f:
        data = json.load(f)

    # 找出所有validation视频的ID
    validation_videos = []
    for video_id, video_data in data['database'].items():
        if video_data.get('subset') == 'validation':
            validation_videos.append(video_id)

    print(f"Found {len(validation_videos)} validation videos")

    # 随机选择100个validation视频改成training
    random.seed(33)  # 设置随机种子以确保结果可重现
    videos_to_change = random.sample(validation_videos, min(100, len(validation_videos)))

    print(f"Changing {len(videos_to_change)} videos from validation to training")

    # 修改选中的视频
    for video_id in videos_to_change:
        data['database'][video_id]['subset'] = 'training'

    # 统计修改后的结果
    train_count = 0
    val_count = 0
    for video_id, video_data in data['database'].items():
        if video_data.get('subset') == 'training':
            train_count += 1
        elif video_data.get('subset') == 'validation':
            val_count += 1

    print(f"After modification:")
    print(f"Training videos: {train_count}")
    print(f"Validation videos: {val_count}")

    # 保存修改后的文件
    with open('dVAR/youcook_json/youcookii_annotations_trainval2.json', 'w') as f:
        json.dump(data, f, indent=4)

    print("File saved successfully!")


def convert2var():
    meta_data = json.load(open('dVAR/youcook_json/youcookii_annotations_trainval2.json', "r"))
    samples = defaultdict(dict)
    for key, value in meta_data['database'].items():
        
        events = [] # change keys in events
        for item in value['annotations']: 
            one_event = {
                'video_id': key,
                'timestamp': item['segment'],
                'sentence': item['sentence'].capitalize(),
                'clip_idx': item['id'],
                'clip_tot': len(value['annotations']),
                'duration': value['duration']
            }
            events.append(one_event)
            
        events = events[:12]
        for i in range(len(events)):
            samples[f'{key}-{i}'] = {
                'hypothesis': i,
                'events': events,
                'split': value['subset'],
                'video_id': key,
                'duration': value['duration']
            }
    
    with open('dVAR/youcook_json/youcookii_trainval_v2.0.json', 'w') as f:
        json.dump(samples, f, indent=4)
    
    print(len(samples), "samples converted to youcook format") # 13829

def prepare_frames(split):
    examples = json.load(open('dVAR/youcook_json/youcookii_trainval_v1.0.json', "r"))
   
    video_info={}
    with open('dVAR/youcook_json/val_duration_totalframe.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:   
            video_info[row[0]]=[float(row[1]), int(row[2])] # duration float, frame_count int   
        
    with open('dVAR/youcook_json/train_duration_totalframe.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:   
            video_info[row[0]]=[float(row[1]), int(row[2])] # duration float, frame_count int   
    
    path_dict = json.load(open('dVAR/youcook_json/trainval_path_dict.json', "r"))
    
    # 使用线程池并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:  # 根据CPU核心数调整
        futures = []
        for exid, sample in examples.items():
            if sample['split'] == split:
                futures.append(executor.submit(
                    process_single_example,
                    exid, sample, split, video_info, path_dict
                ))
        
        # 显示进度条
        for fu in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                fu.result()             
            except Exception as e:
                print(f"[ERROR] {e}")    

def process_single_example(exid, sample, split, video_info, path_dict):
    """多线程处理单个样本的核心函数"""
    output_path = f'/newdata2/xw/youcook2/frames/{exid}.pkl'
    if os.path.exists(output_path):
        print(output_path, "already exists, skipping")
        return  

    # 创建线程本地存储的视频帧缓存
    local_frame_cache = {}
    
    video_ids = set(event['video_id'] for event in sample['events'])

    for vid in video_ids:
        video_path = path_dict[vid]
        if not os.path.exists(video_path):
            video_path = video_path[:-4] + '.mkv' 
        
        if not os.path.exists(video_path):
            video_path = video_path[:-4] + '.webm'
        
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
        frames = frames[start:end]
       
        result_frames.append(frames)
    
    # 保存结果到文件
    with open(output_path, 'wb') as f:
        pickle.dump(result_frames, f)
    
    del local_frame_cache
    del result_frames
    gc.collect()


if __name__ == "__main__":
    # video_val2train()
    # convert2var()
    # prepare_frames('training')  
    YoucookDataset(split='validation', infer=True).__getitem__(19)
    # CausalYoucookDataset(split='training', mode='trainval').__getitem__(9)


   