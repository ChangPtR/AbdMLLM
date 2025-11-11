from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
import os
import torch

class QwenTrainer(Trainer):
    '''For Lora Finetuing'''
    def save_model(self, output_dir=None, _internal_call = False): # finetune whole model
        super().save_model(output_dir, _internal_call)
        params = {}

        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if 'text_projection' in k:
                    params[k] = v.to("cpu")
                if 'unet' in k:
                    params[k] = v.to("cpu")
        
        torch.save(params, os.path.join(output_dir, "trained_params.bin"))
        self.save_state()
      
        


class QwenProjTrainer(Trainer):
    
    def save_model(self, output_dir=None, _internal_call = False): 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # unet_path = os.path.join(output_dir, 'unet')
        # if not os.path.exists(unet_path):
        #     os.makedirs(unet_path, exist_ok=True)

        text_projection_params = {}
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if 'text_projection' in k:
                    text_projection_params[k] = v.to("cpu")
            
        # self.model.unet.save_pretrained(unet_path)
        torch.save(text_projection_params, os.path.join(output_dir, "projector_model.bin"))
        self.save_state()

  
       
        

        
