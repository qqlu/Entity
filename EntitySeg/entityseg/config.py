from detectron2.config import CfgNode as CN

def add_entity_config(cfg):
    """
    Add config for Item.
    """
    ## FCOS Hyper-Parameters
    cfg.MODEL.FCOS = CN()

    # Anchor parameters
    cfg.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.FCOS.NUM_CLASSES = 1
    cfg.MODEL.FCOS.SIZES_OF_INTEREST = [[-1, 64], [64,128], [128,256], [256,512], [512, 100000000]]

    # tower
    cfg.MODEL.FCOS.NUM_CLS_CONVS = 4
    cfg.MODEL.FCOS.NUM_BOX_CONVS = 4
    cfg.MODEL.FCOS.NUM_SHARE_CONVS = 0
    cfg.MODEL.FCOS.CENTER_SAMPLE = True
    cfg.MODEL.FCOS.POS_RADIUS = 1.5
    cfg.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
    cfg.MODEL.FCOS.USE_RELU = True
    cfg.MODEL.FCOS.USE_DEFORMABLE = False
    cfg.MODEL.FCOS.USE_SCALE  = True
    cfg.MODEL.FCOS.TOP_LEVELS = 2
    cfg.MODEL.FCOS.NORM = "GN"

   # loss
    cfg.MODEL.FCOS.PRIOR_PROB    = 0.01
    cfg.MODEL.FCOS.LOSS_ALPHA    = 0.25
    cfg.MODEL.FCOS.LOSS_GAMMA    = 2.0
    cfg.MODEL.FCOS.FB_RATIO      = 4.0
    cfg.MODEL.FCOS.CENTER_SAMPLE = True
    cfg.MODEL.FCOS.YIELD_PROPOSAL = False

    # inference
    cfg.MODEL.FCOS.INFERENCE_TH_TRAIN  = 0.05
    cfg.MODEL.FCOS.INFERENCE_TH_TEST   = 0.05
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TRAIN  = 1000
    cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST   = 1000
    cfg.MODEL.FCOS.NMS_TH              = 0.6
    cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
    cfg.MODEL.FCOS.POST_NMS_TOPK_TEST  = 100
    cfg.MODEL.FCOS.THRESH_WITH_CTR     = False


    ## CONDINST Hyper-Parameters
    cfg.MODEL.CONDINST = CN()
    # the downsampling ratio of the final instance masks to the input image
    cfg.MODEL.CONDINST.MASK_OUT_STRIDE = 4
    cfg.MODEL.CONDINST.MAX_PROPOSALS   = 500
    cfg.MODEL.CONDINST.TRAIN_MAX_PROPOSALS_PER_IMAGE = 120
    cfg.MODEL.CONDINST.LOW_LEVEL_DIMENSION = 16
    cfg.MODEL.CONDINST.CLASS_AGNOSTIC  = False

    cfg.MODEL.CONDINST.MASK_HEAD = CN()
    cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS = 8
    cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS = 3
    cfg.MODEL.CONDINST.MASK_HEAD.USE_FP16 = False
    cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS = False
    cfg.MODEL.CONDINST.MASK_HEAD.CLUSTER_WEIGHT = 1.0
    cfg.MODEL.CONDINST.MASK_HEAD.DYNAMIC = ["111", "110"]
    cfg.MODEL.CONDINST.MASK_HEAD.DYNAMIC_WEIGHT = [1.0, 1.0]

    cfg.MODEL.CONDINST.MASK_BRANCH = CN()
    cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS = 8
    cfg.MODEL.CONDINST.MASK_BRANCH.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.CONDINST.MASK_BRANCH.CHANNELS = 128
    cfg.MODEL.CONDINST.MASK_BRANCH.NORM = "BN"
    cfg.MODEL.CONDINST.MASK_BRANCH.NUM_CONVS = 4
    cfg.MODEL.CONDINST.MASK_BRANCH.SEMANTIC_LOSS_ON = False
    cfg.MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE = False
    ## kernel head
    cfg.MODEL.KERNEL_HEAD    = CN()
    cfg.MODEL.KERNEL_HEAD.NUM_CONVS       = 3
    cfg.MODEL.KERNEL_HEAD.DEFORM          = False
    cfg.MODEL.KERNEL_HEAD.COORD           = True
    cfg.MODEL.KERNEL_HEAD.CONVS_DIM       = 256
    cfg.MODEL.KERNEL_HEAD.NORM            = "GN"

    ## swin transformer backbone
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.PATCH_SIZE = 4
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False

    # # addation
    cfg.MODEL.FPN.TOP_LEVELS = 2

    ## mit former
    cfg.MODEL.MIT_BACKBONE = CN()
    cfg.MODEL.MIT_BACKBONE.NAME = "b0"

    cfg.SOLVER.OPTIMIZER = "sgd"
    cfg.TEST.CLASS_AGNOSTIC = True