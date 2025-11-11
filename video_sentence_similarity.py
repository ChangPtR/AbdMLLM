import torch
import numpy as np
import json
import os
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _get_seq_frames(total_num_frames, desired_num_frames):
    """从视频帧中均匀采样指定数量的帧"""
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


class VideoSentenceDataset(Dataset):
    def __init__(self, split='test'):
        self.split = split
        self.meta_data = json.load(open(f'dVAR/var_json/var_{split}_v1.0.json', "r"))
        self.video_folder = f'dVAR/qwen_data_bound/{split}'
        self.sentences_file = f'resources/var_{split}_infers_detail.json'
        
        # 加载句子数据
        with open(self.sentences_file, 'r', encoding='utf-8') as f:
            self.sentences_data = json.load(f)
        
        # 获取所有有效的视频ID
        self.video_ids = []
        for video_id in self.sentences_data.keys():
            video_path = os.path.join(self.video_folder, f'{video_id}.pkl')

            if os.path.exists(video_path):
                self.video_ids.append(video_id)
        
        print(f'Dataset has {len(self.video_ids)} valid videos')
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        hid = self.meta_data[video_id]['hypothesis']

        with open(f"dVAR/qwen_data_bound/info_{self.split}/{video_id}.pkl", "rb") as f:
            temporal_info = pickle.load(f)
        
        # 加载视频帧
        video_path = os.path.join(self.video_folder, f'{video_id}.pkl')
        with open(video_path, 'rb') as f:
            frames_data = pickle.load(f)
        
        # 处理帧数据 - 将所有clip的帧合并
        st = 0  
        new_frames = []
        for event_idx, num_frames in enumerate(temporal_info):
            if event_idx == hid:
                continue
            for i in range(st,st+num_frames):
                new_frames.append(frames_data[i])

            st += num_frames
        
        # 采样32帧用于表示视频
        if len(new_frames) > 32:
            idx_list = _get_seq_frames(len(new_frames), 32)
            frames_to_select = [new_frames[i] for i in idx_list]
        else:
            frames_to_select = new_frames
        
        # 获取候选句子
        infers = self.sentences_data[video_id]['infers']
        
        return {
            'video_id': video_id,
            'frames': frames_to_select,
            'infers': infers
        }


class VideoSentenceSimilarity:
    def __init__(self, clip_model_path="dVAR/LLM_Weights/clip-vit-large-patch14"):
        """
        初始化CLIP模型用于特征提取和相似度计算
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 加载CLIP模型和处理器
        self.model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_path)
        self.model.eval()
        
    def extract_video_features(self, video_frames):
        """
        从视频帧列表提取CLIP视觉特征
        
        Args:
            video_frames: List[PIL.Image] - 视频帧列表
            
        Returns:
            torch.Tensor: 视频特征 (1, feature_dim)
        """
        if not video_frames:
            return None
        
        # 确保帧是PIL Image对象
        pil_frames = []
        for frame in video_frames:
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame)
            elif not isinstance(frame, Image.Image):
                continue
            pil_frames.append(frame)
        
        if not pil_frames:
            return None
            
        # 处理图像输入
        inputs = self.processor(images=pil_frames, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # 提取每帧的特征
            frame_features = self.model.get_image_features(**inputs)  # (num_frames, feature_dim)
            
            # 对所有帧特征求平均作为视频特征
            video_feature = torch.mean(frame_features, dim=0, keepdim=True)  # (1, feature_dim)
            
        return video_feature
    
    def extract_text_features(self, texts):
        """
        从文本列表提取CLIP文本特征
        
        Args:
            texts: List[str] - 文本列表
            
        Returns:
            torch.Tensor: 文本特征 (num_texts, feature_dim)
        """
        if not texts:
            return None
            
        # 处理文本输入
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)  # (num_texts, feature_dim)
            
        return text_features
    
    def compute_similarity(self, video_feature, text_features):
        """
        计算视频特征和文本特征之间的余弦相似度
        
        Args:
            video_feature: torch.Tensor (1, feature_dim)
            text_features: torch.Tensor (num_texts, feature_dim)
            
        Returns:
            torch.Tensor: 相似度分数 (num_texts,)
        """
        # 归一化特征
        video_feature_norm = F.normalize(video_feature, p=2, dim=1)
        text_features_norm = F.normalize(text_features, p=2, dim=1)
        
        # 计算余弦相似度
        similarities = torch.matmul(text_features_norm, video_feature_norm.T).squeeze()  # (num_texts,)
        
        return similarities
    
    def process_dataset(self, split='test', batch_size=1):
        """
        使用Dataset方式处理视频和句子，计算相似度并筛选前3个最相似的句子
        
        Args:
            split: str - 数据集分割 ('test', 'trainval', etc.)
            batch_size: int - 批处理大小
            
        Returns:
            dict: 结果字典
        """
        # 创建数据集
        dataset = VideoSentenceDataset(split=split)
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        results = {}
        
        with torch.no_grad():
            for data in tqdm(dataset, desc="Processing videos"):
                video_id = data['video_id']
                frames = data['frames']
                infers = data['infers']

                try:
                    # 提取视频特征
                    video_feature = self.extract_video_features(frames)
                    
                    # 提取文本特征
                    text_features = self.extract_text_features(infers)

                    
                    # 计算相似度
                    similarities = self.compute_similarity(video_feature, text_features)
                    
                    # 获取前3个最相似的句子
                    k = min(3, len(infers))
                    top_k_indices = torch.topk(similarities, k=k, largest=True).indices
                    top_k_scores = similarities[top_k_indices]
                    
                    # 保存结果
                    results[video_id] = {
                        'top3_sentences': [infers[idx] for idx in top_k_indices.cpu().numpy()],
                        'top3_scores': top_k_scores.cpu().numpy().tolist(),
                        'top3_indices': top_k_indices.cpu().numpy().tolist(),
                        'total_candidates': len(infers)
                    }    
                except Exception as e:
                    print(f"Error processing video {video_id}: {str(e)}")
                    continue
        
        return results


def process_video_sentence_similarity(split='test'):
    """
    处理视频句子相似度计算的主函数
    
    Args:
        split: str - 数据集分割
    """
    output_file = f"resources/var_{split}_infers_clip_top3.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 初始化处理器
    processor = VideoSentenceSimilarity()
    
    # 使用dataset方式处理
    results = processor.process_dataset(split=split)
    
    # 保存结果
    print(f"Saving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Processing completed! Results saved for {len(results)} videos")
    
    # 打印一些统计信息
    if results:
        avg_candidates = np.mean([v['total_candidates'] for v in results.values()])
        print(f"Average number of candidate sentences per video: {avg_candidates:.2f}")



if __name__ == "__main__":
    split = 'trainval'  # 可以改为 'trainval' 等其他split
    
    # 处理视频句子相似度
    process_video_sentence_similarity(split=split)
