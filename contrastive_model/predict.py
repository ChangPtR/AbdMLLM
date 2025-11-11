from torch.utils.data import DataLoader
from contrastive_model.dataset import VideoTextDataset, data_collator
from contrastive_model.model import CustomModel
from tqdm import tqdm
from contrastive_model.clip_train import Config
import torch
import pickle
import numpy as np
from youcook import CausalYoucookDataset

def infer(split=None,mode=None):
    val_dataset = VideoTextDataset(split=split, mode=mode)
    # val_dataset = CausalYoucookDataset(split, mode)

    val_dataloader = DataLoader(
        val_dataset, batch_size=Config.batch_size, num_workers=4, collate_fn=data_collator
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomModel(Config)
    ckpt = {k.replace('module.', ''): v for k, v in                 
                       torch.load("ckpts/clip_cook1752823755/clip_epoch_7.pt").items()}  # ckpts/clip1747017752/clip_epoch_6.pt
    
    # print(ckpt.keys())
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    print("Ckpt has been loaded!")

    res_dict = {}
    avg_acc = 0
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_dataloader):
            video1 = batch["video1"].to(device)
            video2 = batch["video2"].to(device)
            # video1_mask = batch["video1_mask"].to(device)
            # video2_mask = batch["video2_mask"].to(device)
            label = batch["label"].to(device)
            text = batch["caption"]
            pos = batch["pos"]
            exids = batch["exid"]
            logits = model(video1, video2, text, pos, device).cpu().numpy() 
            for exid, score in zip(exids, logits):
                res_dict[exid] = score

        #     logits = model(video1, video2, text, pos, device)
        #     pred = torch.argmax(logits, dim=1)
        #     acc = torch.sum(pred == label).item() / logits.size(0)
        #     avg_acc += acc    
        # print('avg acc:', avg_acc/len(val_dataloader))

        # pickle.dump(res_dict, open(f'resources/selections/scores_{split}_detail.pkl',"wb"))
        # pickle.dump(res_dict, open(f'resources/selections/scores_blackswan_val_detail.pkl',"wb"))
        pickle.dump(res_dict, open(f'resources/selections/scores_youcookii_{split}_detail.pkl',"wb"))

def view_pkl(split):
    res = pickle.load(open(f'resources/selections/scores_{split}_detail.pkl','rb'))
    value=res['WcnKlchwOnw']
    ids = np.argsort(-value)
    print(ids[:3])

  

                      

if __name__ == "__main__":
    # infer(split='trainval', mode='predict')
    view_pkl(split='test')
            
    # infer(split='validation', mode='predict')

   
    
