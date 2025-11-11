import torch
import torch.nn as nn
from simda_model.unet import UNet3DConditionModel
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from einops import rearrange
import torch.nn.functional as F


class SDModel(nn.Module):
    def __init__(self, model_dir, weight_dtype):
        super().__init__()
        
        self.vae = AutoencoderKL.from_pretrained(
            model_dir, 
            subfolder="vae",
            torch_dtype=weight_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=False
        )

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
    
    def forward(self, target_videos, reasons, video_condition_state=None):
        
        text_inputs = self.clip_tokenizer(
                reasons, max_length=self.clip_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
        text_input_ids = text_inputs.input_ids.to(self.clip_text_encoder.device)
        txt_attention_mask = text_inputs.attention_mask.to(self.clip_text_encoder.device)
        text_embeddings = self.clip_text_encoder(text_input_ids,attention_mask=txt_attention_mask)[0] # 2,77,768
        
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
                                video_condition_state= video_condition_state.to(dtype=torch.bfloat16)
                                ).sample 
       
        target = noise
       
        # txt2img_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        snr = self.compute_snr(timesteps, self.noise_scheduler)
        mse_loss_weights = (
            torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )
        # We first calculate the original loss. Then we mean over the non-batch dimensions and
        # rebalance the sample-wise losses with their respective loss weights.
        # Finally, we take the mean of the rebalanced loss.
        gen_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")
        gen_loss = gen_loss.mean(dim=list(range(1, len(gen_loss.shape)))) * mse_loss_weights
        gen_loss = gen_loss.mean()
        return {'loss': gen_loss}
        
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

        