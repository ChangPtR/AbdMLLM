import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLCausalLMOutputWithPast
)
from transformers import (
    TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor, LogitsProcessorList
)

from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional, Tuple, List, Union
from einops import rearrange,repeat

from tqdm import tqdm
import numpy as np
from simda_model.unet import UNet3DConditionModel
from datetime import datetime


class Qwen2VLForText2ImgConditionalGeneration(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.vae = None 
        self.unet = None
        self.noise_scheduler = None
        self.clip_tokenizer = None
        self.clip_text_encoder = None
            
        # project text embedding from Qwen2VL to CLIP text embedding space
        self.text_projection = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.SiLU(),
            nn.Linear(1024, 768)
        )

        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_videos: Optional[torch.FloatTensor] = None,  # bsz, t, c, h, w 待推理视频gt
        reasons: Optional[List[str]] = None,
        video_condition_state: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        
        # 获取Qwen2VL的输出
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
        )
        txt2img_loss = 0
        text_loss = outputs.loss
        if labels is not None and target_videos is not None:   
            
            text_embeddings = outputs.hidden_states[-1]  # bsz, seq_len, hidden_size
            text_embeddings, _ = self._get_response_embeddings(text_embeddings, labels)
            text_embeddings = self.text_projection(text_embeddings) # bsz, 77, clip_hidden_size
           
           
            video_length = target_videos.shape[1]
            target_videos = rearrange(target_videos, 'b t c h w -> (b t) c h w')  
            latents = self.vae.encode(target_videos.to(dtype=torch.bfloat16)).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
            latents = latents * 0.18215

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
             # Sample a random timestep for each video
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = self.unet(sample=noisy_latents, 
                                   timestep=timesteps, 
                                   encoder_hidden_states=text_embeddings,
                                   video_condition_state=video_condition_state.to(dtype=torch.bfloat16)
                                   ).sample 

            target = noise
       
            snr = self.compute_snr(timesteps, self.noise_scheduler)
            mse_loss_weights = (
                torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            txt2img_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
            txt2img_loss = txt2img_loss.mean(dim=list(range(1, len(txt2img_loss.shape)))) * mse_loss_weights
            txt2img_loss = txt2img_loss.mean() * 5

            if dist.is_initialized() and dist.get_rank() == 0:
                self.logger.debug(
                   "{:.6f}\t{:.6f}\t{}".format(
                    text_loss.item(),
                    txt2img_loss.item(),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                )
            
            outputs.loss = text_loss + txt2img_loss
            
        return outputs

    def l2_loss(self,u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (N, T_I_V_A.txt, D) tensor.
            v: (N, T_I_V_A.txt, D) tensor.
        Returns:
            l1_loss: (N,) tensor of summed L1 loss.
        """
        assert u.shape == v.shape, (u.shape, v.shape)
        return ((u - v) ** 2).sum(dim=-1) ** 0.5

    def compute_snr(self, timesteps, noise_scheduler):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr
    

    def _get_response_embeddings(self, text_embeddings, labels):
        pred_emb = text_embeddings[:, :-1, :]    # shape => (bsz, seq_len-1, dim)
        lab_shift = labels[:, 1:]                # shape => (bsz, seq_len-1)

        response_lengths = (lab_shift != -100).sum(dim=1)  # (bsz,) each data's response length
        max_response_len = 77  # 限制最大长度为77

        bsz, _, hidden_dim = pred_emb.size()
        aligned_embeddings = torch.zeros(bsz, max_response_len, hidden_dim,
                                dtype=pred_emb.dtype,
                                device=pred_emb.device)
        gen_attention_mask = torch.zeros(bsz, max_response_len, dtype=torch.bool, device=pred_emb.device)
       
        for i in range(bsz):
            response_mask = (lab_shift[i] != -100)  # seq_len
            res_len = min(response_lengths[i].item(), max_response_len)  # 确保不超过77
            aligned_embeddings[i, :res_len] = pred_emb[i][response_mask][:res_len]  # 截断超过77的部分
            gen_attention_mask[i, :res_len] = 1

        return aligned_embeddings, gen_attention_mask

    def init_unet_modules(self, model_dir, pretrained_unet_dir=None, weight_dtype=torch.bfloat16):
        self.vae = AutoencoderKL.from_pretrained(
            model_dir, 
            subfolder="vae",
            use_safetensors=True,
            low_cpu_mem_usage=False,
            torch_dtype=weight_dtype
        )
      
        if pretrained_unet_dir is not None:
            self.unet = UNet3DConditionModel.from_pretrained(
                pretrained_unet_dir,
                torch_dtype=weight_dtype
            ) 
        else:
            self.unet = UNet3DConditionModel.from_pretrained_2d(
                model_dir,
                subfolder="unet",
                torch_dtype=weight_dtype    
            ) 

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_dir, subfolder="scheduler") 

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        self.clip_text_encoder = CLIPTextModel.from_pretrained(model_dir, 
                                                               subfolder="text_encoder",
                                                               torch_dtype=weight_dtype)


    def init_unet_only(self, model_dir, weight_dtype):
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            model_dir, 
            subfolder="unet",
            torch_dtype=weight_dtype
        ) 