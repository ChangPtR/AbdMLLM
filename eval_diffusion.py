import torch
from safetensors import safe_open
import os
from tqdm import tqdm
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
import torch.nn.functional as F
from collections import defaultdict
import json
from util.data_util import _get_seq_frames
from models.unet_3d_condition import UNet3DConditionModel
from models.pipeline_mavin import MAVINPipeline
from diffusers import DPMSolverMultistepScheduler
from ms_t2v.finetune_ms import VARDataset, DataCollatorForSupervisedDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from diffusers.utils import export_to_video
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms


def cal_clip_score():
    gt_path = '/data/processed/VAR_Dataset/qwen_data_bound/gt_test'
    pred_path = 'arg_imgs/simda_ft2'
    text_condition = {item['id']: item for item in json.load(open('/home/cby/finetune-Qwen2-VL/resources/var_test_list_msg0.json', "r"))}
    video_length=16

    clip_model = CLIPModel.from_pretrained("/data/LLM_Weights/clip-vit-large-patch14").to('cuda')
    clip_processor = CLIPProcessor.from_pretrained("/data/LLM_Weights/clip-vit-large-patch14")
    clipsim_dict = defaultdict()
    clipsim_img = 0
    clipsim_text = 0
    num = len(os.listdir(pred_path))
    for exid in tqdm(os.listdir(pred_path)):
        pred_frames_path = [os.path.join(pred_path, exid, f) for f in os.listdir(os.path.join(pred_path, exid)) if f.endswith('.png')]
        pred_frames_path.sort()  # 确保帧按顺序排列
        pred_frames = [Image.open(frame) for frame in pred_frames_path]
        # print(pred_frames[0].size, gt_frames[0].size)
        target_size = pred_frames[0].size  # 获取预测帧的尺寸 (宽度, 高度)
        
        gt_frames = pickle.load(open(os.path.join(gt_path, f'{exid}.pkl'), 'rb'))
        gt_ids = _get_seq_frames(len(gt_frames), video_length)
        gt_frames = [gt_frames[i].resize(target_size) for i in gt_ids]

        gt_inputs = clip_processor(images=gt_frames, return_tensors="pt").to("cuda")
        pred_inputs = clip_processor(images=pred_frames, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            gt_features = clip_model.get_image_features(**gt_inputs)
            pred_features = clip_model.get_image_features(**pred_inputs)
        
        similarity = F.cosine_similarity(gt_features, pred_features, dim=1)
        similarity1 = similarity.mean().item()
        clipsim_img += similarity1  

        text = text_condition[exid]['messages'][2]['content'][0]['text']
        text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to('cuda')
        with torch.no_grad():
            text_outputs = clip_model.get_text_features(**text_inputs)  # (1, dim)
        text_feature = text_outputs[0].unsqueeze(0)

        similarity = F.cosine_similarity(text_feature, pred_features, dim=1)
        similarity2 = similarity.mean().item()
        clipsim_text += similarity2

        clipsim_dict[exid] = [similarity1, similarity2]
    
    clipsim_img /= num
    clipsim_text /= num
    print(f"CLIP similarity: {clipsim_img:.4f}, {clipsim_text:.4f}")
    json.dump(clipsim_dict, open('resources/gen_score/clipsim_ft2.json', 'w'))


class VARDatasetTest(Dataset):

    def __init__(self, split):
        super().__init__()
        self.split = split
        self.meta_data = json.load(open(f'dVAR/var_json/var_{split}_v1.0.json', "r"))    
        self.samples = []
        for k,v in self.meta_data.items():
            self.samples.append({
                'exid': k,
                **v
            })
                
        print(f'vardataset has {len(self.samples)} samples')

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR), 
            transforms.CenterCrop(256),  
            transforms.ToTensor(),  
        ])
        self.max_event_frames = 16
        
        self.condition_json = json.load(open('dVAR/sd_train_data/var_trainval_ids.json', "r"))
        self.topk = 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        exid = sample['exid']
        clip_idx = sample["hypothesis"]

        with open(f"/newdata1/xw/qwen_data_all/{self.split}/{exid}.pkl", "rb") as f:
            frames = pickle.load(f)
       
        gt_frames = frames[clip_idx]
        gt_ids = _get_seq_frames(len(gt_frames), self.max_event_frames)
        gt_frames = [(self.transform(gt_frames[i])*2 - 1) for i in gt_ids]

        # get condition frames from other video clips
        frames_to_select = []
        for i in range(len(frames)):
            if i == clip_idx:
                continue
            frames_to_select.extend(frames[i])
        
        idx_list = _get_seq_frames(len(frames_to_select), 32) # represent oberved video with 32 frames
        select_ids = self.condition_json[f'{exid}-{clip_idx}']['topk_idx'][:self.topk]
        final_ids = [idx_list[i] for i in select_ids]
        condition_frames = [frames_to_select[i] for i in final_ids]

        # inference data
        if clip_idx == 0:
            pos = 'BEFORE' 
            boundary_list = [(self.transform(frames[1][0])*2 - 1)]*2
        elif clip_idx == len(frames) - 1:
            pos = 'AFTER'
            boundary_list = [(self.transform(frames[-2][-1])*2 - 1)]*2
        else:
            pos = 'BETWEEN'
            boundary_list = [(self.transform(frames[clip_idx-1][-1])*2 - 1), (self.transform(frames[clip_idx+1][0])*2 - 1)]

        return {
            'target_videos': gt_frames,
            'reason': sample['events'][clip_idx]['sentence'],
            'condition_frames': condition_frames,
            'pos': pos,
            'boundary_frame': boundary_list,
            'id': f'{exid}-{clip_idx}'
        }

def run_generation():

    config = OmegaConf.load("ms_t2v/connection.yaml")
    unet_dir = "ckpts/ms_5e-6_videocond2/checkpoint-6483"

    user_config = OmegaConf.to_container(config.user_model_config, resolve=True)
    # unet = UNet3DConditionModel.from_pretrained(config.pretrained_model_path, 
    #                                             subfolder="unet", 
    #                                             user_model_config=user_config,
    #                                             low_cpu_mem_usage=False, torch_dtype=torch.float16).to('cuda')
    unet = UNet3DConditionModel.from_pretrained(unet_dir, torch_dtype=torch.float16).to('cuda')

    pipe = MAVINPipeline.from_pretrained(config.pretrained_model_path, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    clip_processor = CLIPProcessor.from_pretrained('dVAR/LLM_Weights/clip-vit-large-patch14')
    clip_vision_encoder = CLIPVisionModel.from_pretrained('dVAR/LLM_Weights/clip-vit-large-patch14').to('cuda')

    testset = VARDatasetTest(split='trainval')
    testset.samples = testset.samples[471:476]
    testloader = DataLoader(testset, batch_size=4, shuffle=False, collate_fn=DataCollatorForSupervisedDataset())

    save_dir = 'arg_imgs/ms_videocond2_3epoch_latent'
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.inference_mode():
        vae = pipe.vae
        for sample in tqdm(testloader):
            exids = sample['ids']
            output_texts = sample['reasons']
            print(exids, output_texts)

            # Inference video 
            generator = torch.Generator(device='cuda')
            generator.manual_seed(42)

            boundary_frames = sample['boundary_frames'] # (b, 2, c, h, w)
            boundary_frames = torch.cat([boundary_frames[:, 0:1,:,:,:].repeat(1, 8, 1, 1, 1), 
                                        boundary_frames[:, 1:2,:, :, :].repeat(1, 8, 1, 1, 1)], dim=1) 
            video_length = boundary_frames.shape[1]
         
            boundary_frames = rearrange(boundary_frames, "b f c h w -> (b f) c h w")
            latents = vae.encode(boundary_frames.to(vae.device, dtype=torch.float16)).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
            latents = latents * 0.18215
            print(latents.shape, latents.device, latents.dtype)

            video_conditions = sample['video_conditions']
            bsz, fnum = len(video_conditions), len(video_conditions[0])
            flat_images = [img for sample in video_conditions for img in sample] 
            video_inputs = clip_processor(images=flat_images, return_tensors="pt").pixel_values
            video_condition_state = clip_vision_encoder(video_inputs.to(latents.device))[0]
            video_condition_state = rearrange(video_condition_state, "(b f) s d -> b (f s) d", b=bsz, f=fnum)   
            # print(video_condition_state.shape)

            if config.frameinit_kwargs.enable:
                pipe.init_filter(
                    filter_shape=latents.shape, # b c f h w
                    filter_params=config.frameinit_kwargs.filter_params,
                )

            batch_videos = pipe(output_texts,
                                width=256,
                                height=256,
                                num_frames=16,
                                num_inference_steps=100,
                                guidance_scale=9.0,
                                generator=generator,
                                starter_latent=latents,
                                # latents=latents,
                                encoder_image_states=video_condition_state.to(dtype=latents.dtype),
                                shared_noise=True,
                                pos=sample['poses'],
                                use_frameinit=config.frameinit_kwargs.enable,
                                use_gfm=True,
                                noise_level=config.frameinit_kwargs.noise_level).frames

            for exid, video_frames in zip(exids, batch_videos):
                print(video_frames.shape)
                out_file = f'{save_dir}/{exid}.mp4'
                export_to_video(video_frames, out_file, 4)

if __name__ == "__main__":
    # cal_clip_score()
    run_generation()