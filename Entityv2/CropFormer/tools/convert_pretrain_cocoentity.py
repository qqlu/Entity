import torch
import pdb

infos = torch.load("/group/20027/gavinqi/model/entityv2_50ep_with_coco_same_epoch/model_final.pth")
weights = infos["model"]
new_weights = {}
for key, value in weights.items():
    print(key)
    if 'sem_seg_head.pixel_decoder.pixel_decoder' in key:
        pdb.set_trace()
        _, new_key_2 = key.split("sem_seg_head.pixel_decoder.pixel_decoder")
        new_key = "sem_seg_head.pixel_decoder" + new_key_2
        new_weights[new_key]=value
        print(new_key)
    else:
        new_weights[key]=value
infos["model"] = new_weights
torch.save(infos, "/group/20027/gavinqi/model/entityv2_50ep_with_coco_same_epoch/model_final_new_mask2former.pth")

# pdb.set_trace()