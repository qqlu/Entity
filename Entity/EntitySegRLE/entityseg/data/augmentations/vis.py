import copy
import torch
import torchvision


def _vis(tensor, boxes, iteration=0, prex=''):
    
    a = torch.clone(tensor)
    width = 3
    for box in boxes:
        y1, x1, y2, x2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        patch = copy.deepcopy(a[:,x1+width:x2-width, y1+width:y2-width])
        a[:,x1:x2,y1:y2] = torch.Tensor([1, 0, 0]).reshape(1, 3, 1, 1)
        a[:,x1+width:x2-width, y1+width:y2-width] = patch

    inv_idx = torch.arange(a.shape[0]-1,-1,-1).long().tolist()
    torchvision.utils.save_image(a[inv_idx].float(), '%s_tensor_%d.jpg'%(prex, iteration), normalize=True)
