import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset

from models.network.crm import CRMNet
from models.sobel_op import SobelComputer
from dataset import OnlineTransformDataset_crm as OnlineTransformDataset
from util.logger import BoardLogger
from util.model_saver import ModelSaver
from util.hyper_para import HyperParameters
from util.log_integrator import Integrator
from util.metrics_compute_crm import compute_loss_and_metrics, iou_hooks_to_be_used
from util.image_saver_crm import vis_prediction

import time
import os
import datetime

torch.backends.cudnn.benchmark = True

# Parse command line arguments
para = HyperParameters()
para.parse()

# Logging
if para['id'].lower() != 'null':
    long_id = '%s_%s' % (para['id'],datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
else:
    long_id = None
logger = BoardLogger(long_id)
logger.log_string('hyperpara', str(para))

print('CUDA Device count: ', torch.cuda.device_count())

# Construct model
model = CRMNet(backend='resnet50')
model = nn.DataParallel(
        model.cuda(), device_ids=[0,1]
    )

if para['load'] is not None:
    model.load_state_dict(torch.load(para['load']))
optimizer = optim.Adam(model.parameters(), lr=para['lr'], weight_decay=para['weight_decay'])


duts_tr_dir = os.path.join('data', 'DUTS-TR')
duts_te_dir = os.path.join('data', 'DUTS-TE')
ecssd_dir = os.path.join('data', 'ecssd')
msra_dir = os.path.join('data', 'MSRA_10K')

fss_dataset = OnlineTransformDataset(os.path.join('data', 'fss'), method=0, perturb=True)
duts_tr_dataset = OnlineTransformDataset(duts_tr_dir, method=1, perturb=True)
duts_te_dataset = OnlineTransformDataset(duts_te_dir, method=1, perturb=True)
ecssd_dataset = OnlineTransformDataset(ecssd_dir, method=1, perturb=True)
msra_dataset = OnlineTransformDataset(msra_dir, method=1, perturb=True)

print('FSS dataset size: ', len(fss_dataset))
print('DUTS-TR dataset size: ', len(duts_tr_dataset))
print('DUTS-TE dataset size: ', len(duts_te_dataset))
print('ECSSD dataset size: ', len(ecssd_dataset))
print('MSRA-10K dataset size: ', len(msra_dataset))

train_dataset = ConcatDataset([fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset, msra_dataset])

print('Total training size: ', len(train_dataset))

# For randomness: https://github.com/pytorch/pytorch/issues/5059
def worker_init_fn(worker_id): 
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Dataloaders, multi-process data loading
train_loader = DataLoader(train_dataset, para['batch_size'], shuffle=True, num_workers=8,
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)

sobel_compute = SobelComputer()

# Learning rate decay scheduling
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, para['steps'], para['gamma'])

saver = ModelSaver(long_id)
report_interval = 50
save_im_interval = 800
memory_chunk = 50176

total_epoch = int(para['iterations']/len(train_loader) + 0.5)
print('Actual training epoch: ', total_epoch)

train_integrator = Integrator(logger)
train_integrator.add_hook(iou_hooks_to_be_used)
total_iter = 0
last_time = 0
for e in range(total_epoch):
    np.random.seed() # reset seed
    epoch_start_time = time.time()

    # Train loop
    model = model.train()
    for im, seg, gt, crm_data in train_loader:
        im, seg, gt = im.cuda(), seg.cuda(), gt.cuda() # [12, 3, 224, 224] [12, 1, 224, 224] [12, 1, 224, 224]
        for k, v in crm_data.items():
            crm_data[k] = v.cuda()

        total_iter += 1
        if total_iter % 5000 == 0:
            saver.save_model(model, total_iter)

        images = {}
        for i in range(0, seg.shape[-2]*seg.shape[-1], memory_chunk):
            chunk_images = model(im, seg, coord=crm_data['coord'][:, i:i+memory_chunk, :], cell=crm_data['cell'][:, i:i+memory_chunk, :])
            if 'pred_224' not in images.keys():
                images = chunk_images
            else:
                for key in images.keys():
                    images[key] = torch.cat((images[key], chunk_images[key]), axis=1)
        for key in images.keys():
            images[key] = images[key].view(images[key].shape[0], images[key].shape[1]//(seg.shape[-2]*seg.shape[-1]), *seg.shape[-2:])

        images['im'] = im
        images['seg'] = seg
        images['gt'] = gt
        sobel_compute.compute_edges(images)

        loss_and_metrics = compute_loss_and_metrics(images, para)
        train_integrator.add_dict(loss_and_metrics)

        optimizer.zero_grad()
        (loss_and_metrics['total_loss']).backward()
        optimizer.step()

        if total_iter % report_interval == 0:
            logger.log_scalar('train/lr', scheduler.get_lr()[0], total_iter)
            train_integrator.finalize('train', total_iter)
            train_integrator.reset_except_hooks()

        # Need to put step AFTER get_lr() for correct logging, see issue #22107 in PyTorch
        scheduler.step()

        if total_iter % save_im_interval == 0:
            predict_vis = vis_prediction(images)
            logger.log_cv2('train/predict', predict_vis, total_iter)

# Final save!
saver.save_model(model, total_iter)
