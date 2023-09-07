## Mask2Former Demo

We provide a command line tool to run a simple demo of builtin configs.
The usage is explained in [GETTING_STARTED.md](../GETTING_STARTED.md).

## 
python3 projects/Mask2Former/demo_cropformer/demo_from_dirs.py --config-file projects/Mask2Former/configs/entityv2/entity_segmentation/cropformer_swin_large_3x.yaml --input /group/20018/gavinqi/demo_images/ft_local/*.jpeg --output /group/20027/gavinqi/debug_demo/ --opts MODEL.WEIGHTS /group/20027/gavinqi/model/TPAMI_entityseg_cropformer_swin_large_cocopretrain_debugv3_has_crop_modify2d_add3d_split_pos2d3d_shared_structure_2D05_hasflip_all_3x/model_final.pth

python3 projects/Mask2Former/demo_cropformer/demo_from_dirs.py --config-file projects/Mask2Former/configs/entityv2/entity_segmentation/cropformer_swin_large_3x.yaml --input /group/20027/gavinqi/data/ft_local/artistic_images/*.jp* --output /group/20027/gavinqi/data/ft_local/artistic_images_seg --opts MODEL.WEIGHTS /group/20027/gavinqi/model/TPAMI_entityseg_cropformer_swin_large_cocopretrain_debugv3_has_crop_modify2d_add3d_split_pos2d3d_shared_structure_2D05_hasflip_all_3x/model_final.pth

## 
python3 projects/Mask2Former/demo_cropformer/demo_from_dirs.py --config-file projects/Mask2Former/configs/coco_person/cropformer_swin_large_3x_noise_000_100_200.yaml --input /group/20018/gavinqi/data/ft_local/100m_crop_sample/*.jpg --output /group/20027/gavinqi/100m_vis/ --opts MODEL.WEIGHTS /group/20027/gavinqi/model/coco_person_noise_000_100_200/model_final.pth

## 
python3 projects/Mask2Former/demo_cropformer/demo_from_txt_only_bimask.py --config-file projects/Mask2Former/configs/coco_person/cropformer_swin_large_3x_noise_000_100_200.yaml --input /group/20018/gavinqi/data/ft_local/100m_crop_sample.txt --output /group/20027/gavinqi/100m_vis/ --thread-id 0 --thread-num 1 --opts MODEL.WEIGHTS /group/20027/gavinqi/model/coco_person_noise_000_100_200/model_final.pth


### diffusion
python3 projects/Mask2Former/demo_cropformer/demo_from_diffusion_images.py --config-file projects/Mask2Former/configs/entityv2/entity_segmentation/cropformer_swin_large_3x.yaml --output /group/20027/gavinqi/diffusion_vis_two_entity --opts MODEL.WEIGHTS /group/20027/gavinqi/model/TPAMI_entityseg_cropformer_swin_large_cocopretrain_debugv3_has_crop_modify2d_add3d_split_pos2d3d_shared_structure_2D05_hasflip_all_3x/model_final.pth