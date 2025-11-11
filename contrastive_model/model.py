import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class Projection(nn.Module):
    '''project text features to the casual space'''
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class VideoEncoder(nn.Module):
    '''project video features to the casual space'''
    def __init__(self,  d_in: int, d_out: int) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_in,
            nhead=8,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_in, d_out)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, features, mask=None):
        vec = self.encoder(features, src_key_padding_mask=mask)
        vec = self.fc(vec)
        projected_vec = self.pool(vec.transpose(1, 2)).squeeze(-1)
        return projected_vec

class TextEncoder(nn.Module):
    def __init__(self, model_dir: str, d_out: int) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.base = AutoModel.from_pretrained(model_dir) # dim=768
        self.projection = Projection(self.base.config.hidden_size, d_out)
 
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, text, device):
        input_ids = self.tokenizer(text,
                            return_tensors="pt",
                            padding="longest",
                            max_length=512,
                            truncation=True).input_ids
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id).to(device)
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id).to(device)
        
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)[0]
        out = out[:, 0, :]  # get CLS token output
        projected_vec = self.projection(out)
        return projected_vec

class CustomModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.vision_encoder = VideoEncoder(config.video_feature_dim, config.embed_dim)  
        self.caption_encoder = TextEncoder(config.text_encoder_path, config.embed_dim)

    def forward(self, video1, video2, texts, pos, device):
        batch_size = video1.size(0)
        cap_num = len(texts[0]) # text[i] has n captions

        # video_vec1 = self.vision_encoder(video1, video1_mask) 
        # video_vec2 = self.vision_encoder(video2, video2_mask)
        video_vec1 = self.vision_encoder(video1) 
        video_vec2 = self.vision_encoder(video2)

        # flat the +- samples for text encoder
        flat_texts = [t for vid_texts in texts for t in vid_texts]  # (batch_size,n) => batch_size*n sentences
        txt_vec = self.caption_encoder(flat_texts, device)  # (batch_size*n, embed_dim)
        txt_vec = txt_vec.view(batch_size, cap_num, -1)  # (batch_size, n, embed_dim)

        # L2 normalize
        video_vec1 = F.normalize(video_vec1, p=2, dim=-1)
        video_vec2 = F.normalize(video_vec2, p=2, dim=-1)
        txt_vec = F.normalize(txt_vec, p=2, dim=-1)
        
        # position mask: each sample in the batch is BEFORE/BETWEEN/AFTER
        mask_be = torch.tensor([p in ['BEFORE', 'BETWEEN'] for p in pos], dtype = torch.bool, device = device)
        mask_af = torch.tensor([p == 'AFTER' for p in pos], dtype = torch.bool, device = device)
        
        logits = torch.zeros(batch_size, cap_num, device = device)

        # BEFORE/BETWEEN
        if mask_be.any():
            merged = video_vec1[mask_be].unsqueeze(1) + txt_vec[mask_be] # broadcast: (bs,1,dim) + (bs,n,dim) = (bs,n,dim)
            sim = torch.bmm(merged, video_vec2[mask_be].unsqueeze(2)).squeeze() # (bs,n,dim) * (bs,dim,1) = (bs,n,1) -> (bs,n)
            logits[mask_be] = sim

        # AFTER
        if mask_af.any():
            merged_video = (video_vec1[mask_af] + video_vec2[mask_af]).unsqueeze(1) # (bs,1,dim)
            sim = torch.bmm(txt_vec[mask_af], merged_video.transpose(1,2)).squeeze() # (bs,n,dim) * (bs,dim,1) = (bs,n,1) -> (bs,n)
            logits[mask_af] = sim
        
        return logits # (bs,cls_num=n)
    


# model = TextEncoder("/data/LLM_Weights/bert-base-uncased", 512)
# print(model.base.config.hidden_size)