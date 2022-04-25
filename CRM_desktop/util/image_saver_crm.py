import cv2
import numpy as np

import torchvision.transforms as transforms

inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

inv_seg_trans = transforms.Normalize(
    mean=[-0.5/0.5],
    std=[1/0.5])

def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def tensor_to_np_float(image):
    image_np = image.numpy().astype('float32')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def transpose_np(x):
    return np.transpose(x, [1,2,0])

def tensor_to_gray_im(x):
    x = detach_to_cpu(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_seg(x):
    x = detach_to_cpu(x)
    x = inv_seg_trans(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

def tensor_to_im(x):
    x = detach_to_cpu(x)
    x = inv_im_trans(x)
    x = tensor_to_numpy(x)
    x = transpose_np(x)
    return x

# Predefined key <-> caption dict
key_captions = {
    'im': 'Image', 
    'gt': 'GT', 
    'seg': 'Input', 
    'error_map': 'Error map',
}
for k in ['28', '56', '224']:
    key_captions['pred_' + k] = 'Ours-%sx%s' % (k, k)
    key_captions['pred_' + k + '_overlay'] = '%sx%s' % (k, k)

"""
Return an image array with captions
keys in dictionary will be used as caption if not provided
values should contain lists of cv2 images
"""
def get_image_array(images, grid_shape, captions={}):
    w, h = grid_shape
    cate_counts = len(images)
    rows_counts = len(next(iter(images.values())))

    font = cv2.FONT_HERSHEY_SIMPLEX

    output_image = np.zeros([h*(rows_counts+1), w*cate_counts, 3], dtype=np.uint8)
    col_cnt = 0
    for k, v in images.items():

        # Default as key value itself
        caption = captions.get(k, k)

        # Handles new line character
        y0, dy = h-10-len(caption.split('\n'))*40, 40
        for i, line in enumerate(caption.split('\n')):
            y = y0 + i*dy
            cv2.putText(output_image, line, (col_cnt*w, y),
                     font, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Put images
        for row_cnt, img in enumerate(v):
            im_shape = img.shape
            if len(im_shape) == 2:
                img = img[..., np.newaxis]

            img = (img * 255).astype('uint8')

            output_image[(row_cnt+1)*h:(row_cnt+2)*h,
                         col_cnt*w:(col_cnt+1)*w, :] = img
            
        col_cnt += 1

    return output_image

"""
Create an image array, transform each image separately as needed
Will only put images in req_keys
"""
def pool_images(images, req_keys, row_cnt=10):
    req_images = {}

    def base_transform(im):
        im = tensor_to_np_float(im)
        im = im.transpose((1, 2, 0))

        # Resize
        if im.shape[1] != 224:
            im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_NEAREST)

        if len(im.shape) == 2:
            im = im[..., np.newaxis]

        return im

    second_pass_keys = []
    for k in req_keys:

        if 'overlay' in k: 
            # Run overlay in the second pass, skip for now
            second_pass_keys.append(k)

            # Make sure the base key information is transformed
            base_key = k.replace('_overlay', '')
            if base_key in req_keys:
                continue
            else:
                k = base_key

        req_images[k] = []

        images[k] = detach_to_cpu(images[k])
        for i in range(min(row_cnt, len(images[k]))):

            im = images[k][i]

            # Handles inverse transform
            if k in ['im']:
                im = inv_im_trans(images[k][i])
            elif k in ['seg']:
                im = inv_seg_trans(images[k][i])

            # Now we are all numpy array
            im = base_transform(im)

            req_images[k].append(im)

    # Handle overlay images in the second pass
    for k in second_pass_keys:
        req_images[k] = []
        base_key = k.replace('_overlay', '')
        for i in range(min(row_cnt, len(images[base_key]))):

            # If overlay
            im = req_images[base_key][i]
            raw = req_images['im'][i]

            im = im.clip(0, 1)

            # Just red overlay
            im = (raw*0.5 + 0.5 * (raw * (1-im) 
                    + im * (np.array([1,0,0],dtype=np.float32)
                    .reshape([1,1,3]))))
            
            req_images[k].append(im)
    
    # Remove all temp items
    output_images = {}
    for k in req_keys:
        output_images[k] = req_images[k]

    return get_image_array(output_images, (224, 224), key_captions)

# Return cv2 image, directly usable for saving
def vis_prediction(images):

    keys = ['im', 'seg', 'gt', 'pred_224', 'pred_224_overlay'] # 'pred_28', 'pred_28_2', 'pred_56', 'pred_28_3', 'pred_56_2', 

    return pool_images(images, keys)
