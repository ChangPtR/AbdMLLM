from torch.utils.data import Dataset
import json
import math
import os
import numpy as np
from util.data_util import _get_seq_frames
from torch.utils.data import DataLoader
import torch

class VideoTextDataset(Dataset):
    def __init__(self, split, mode):
        '''
        mode=trainval: positive_texts, negative_texts
        mode=predict: possible infers
        '''
        self.mode = mode
        self.split = split
        self.video_feature_dir = 'dVAR/video_feature'
        self.duration_file = 'dVAR/var_json/var_video_duration_v1.0.csv'
        self.frame_to_second = self._load_duration() # 每个特征=多少秒
        self.max_v_len = 300

        self.positive_texts = json.load(open(f'dVAR/var_json/var_{split}_v1.0.json', "r"))
        if mode == 'trainval':       
            if split == 'trainval':
                negative_texts = json.load(open(f'resources/var_{split}_negative100.json', "r"))    
            elif split == 'test':
                negative_texts = json.load(open(f'resources/var_{split}_infers_detail.json', "r")) 
        elif mode == 'predict':
            negative_texts = json.load(open(f'resources/var_{split}_infers_detail.json', "r"))          
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
            pos_text = self.positive_texts[exid]["events"][hid]["sentence"] 
            if self.split == 'trainval':
                all_texts = neg_dict['neg'][:100]
            elif self.split == 'test':
                all_texts = neg_dict['infers']
            idx = np.random.randint(0, len(all_texts), dtype=np.int64)
            all_texts.insert(idx, pos_text) # insert positive text for eval
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
        sampling_sec = 0.5  # hard coded, only support 0.5
        with open(self.duration_file, "r") as f:
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_to_second[vid_name] = float(vid_dur)*math.ceil(float(vid_frame)*1./float(vid_dur)*sampling_sec)*1./float(vid_frame)
        return frame_to_second # ~0.5
    
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
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len-1)
        st = min(st, ed-1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed
    
    def pad_video(self, video, max_v_len):
        '''
        torch TransformerEncoder src_key_padding_mask:
        False means there is no padding token there (so yes use that value in the transformer forward pass) 
        and a True means that there is a padding token (so masked it out so the transformer pass forward does not get affected).
        '''
        # if video.shape[0] < max_v_len:
        #     pad_size = max_v_len - video.shape[0]
        #     new_video = np.concatenate([video, np.zeros((pad_size, video.shape[1]))], axis=0)
        #     mask = np.concatenate([np.zeros(video.shape[0]), np.ones(pad_size)])
        # else:
        #     new_video = video
        #     if video.shape[0] > max_v_len:
        #         idx_list = _get_seq_frames(len(video), max_v_len)
        #         new_video = video[idx_list]
        #     mask = np.zeros(max_v_len, dtype=np.int64)
        # return new_video, mask
        new_video = video
        idx_list = _get_seq_frames(len(video), max_v_len)
        new_video = video[idx_list]
        return new_video

def data_collator(instances):
    video1_list = [torch.tensor(instance["video1"]) for instance in instances]
    video2_list = [torch.tensor(instance["video2"]) for instance in instances]
    # video1_mask_list = [torch.tensor(instance["video1_mask"]) for instance in instances]
    # video2_mask_list = [torch.tensor(instance["video2_mask"]) for instance in instances]
    label_list = [instance["label"] for instance in instances]
    caption_list = [instance["caption"] for instance in instances]
    pos_list = [instance["pos"] for instance in instances]
    exid_list = [instance["exid"] for instance in instances]

    return {
        "video1": torch.stack(video1_list),
        "video2": torch.stack(video2_list),
        # "video1_mask": torch.stack(video1_mask_list),
        # "video2_mask": torch.stack(video2_mask_list),
        "label": torch.tensor(label_list),
        "caption": caption_list,
        "pos": pos_list,
        "exid": exid_list
    }


clip_dataset = VideoTextDataset(split="trainval", mode='trainval')
clip_dataloader = DataLoader(
        clip_dataset, batch_size=1, num_workers=4, collate_fn=data_collator
)
for i, batch in enumerate(clip_dataloader):
    pass
