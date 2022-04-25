import torchvision.transforms as transforms

import os

from torch.utils.tensorboard import SummaryWriter
# import git
import warnings

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def fix_width_trunc(x):
    return ('{:.9s}'.format('{:0.9f}'.format(x)))

class BoardLogger:
    def __init__(self, id):

        if id is None:
            self.no_log = True
            warnings.warn('Logging has been disbaled.')
        else:
            self.no_log = False

            self.inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

            self.inv_seg_trans = transforms.Normalize(
                mean=[-0.5/0.5],
                std=[1/0.5])

            log_path = os.path.join('.', 'log', '%s' % id)
            self.logger = SummaryWriter(log_path)

        # repo = git.Repo(".")
        # self.log_string('git', str(repo.active_branch) + ' ' + str(repo.head.commit.hexsha))

    def log_scalar(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        self.logger.add_scalar(tag, x, step)

    def log_metrics(self, l1_tag, l2_tag, val, step, f=None):
        tag = l1_tag + '/' + l2_tag
        text = 'It {:8d} [{:5s}] [{:19s}]: {:s}'.format(step, l1_tag.upper(), l2_tag, fix_width_trunc(val))
        print(text)
        if f is not None:
            f.write(text + '\n')
            f.flush()
        self.log_scalar(tag, val, step)

    def log_im(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = self.inv_im_trans(x)
        x = tensor_to_numpy(x)
        self.logger.add_image(tag, x, step)

    def log_cv2(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = x.transpose((2, 0, 1))
        self.logger.add_image(tag, x, step)

    def log_seg(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = self.inv_seg_trans(x)
        x = tensor_to_numpy(x)
        self.logger.add_image(tag, x, step)

    def log_gray(self, tag, x, step):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        x = detach_to_cpu(x)
        x = tensor_to_numpy(x)
        self.logger.add_image(tag, x, step)

    def log_string(self, tag, x):
        print(tag, x)
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        self.logger.add_text(tag, x)

    def log_total(self, tag, im, gt, seg, pred, step):
        
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        
        row_cnt = min(10, im.shape[0])
        w = im.shape[2]
        h = im.shape[3]
        
        output_image = np.zeros([3, w*row_cnt, h*5], dtype=np.uint8)
        
        for i in range(row_cnt):
            im_ = tensor_to_numpy(self.inv_im_trans(detach_to_cpu(im[i])))
            gt_ = tensor_to_numpy(detach_to_cpu(gt[i]))
            seg_ = tensor_to_numpy(self.inv_seg_trans(detach_to_cpu(seg[i])))
            pred_ = tensor_to_numpy(detach_to_cpu(pred[i]))
            
            output_image[:, i * w : (i+1) * w, 0 : h] = im_
            output_image[:, i * w : (i+1) * w, h : 2*h] = gt_
            output_image[:, i * w : (i+1) * w, 2*h : 3*h] = seg_
            output_image[:, i * w : (i+1) * w, 3*h : 4*h] = pred_
            output_image[:, i * w : (i+1) * w, 4*h : 5*h] = im_*0.5 + 0.5 * (im_ * (1-(pred_/255)) + (pred_/255) * (np.array([255,0,0],dtype=np.uint8).reshape([1,3,1,1])))
            
        self.logger.add_image(tag, output_image, step)
