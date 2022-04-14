'''
Author: Catherine Xiong
Date: 2022-03-03 09:50:59
LastEditTime: 2022-03-16 00:33:25
LastEditors: Catherine Xiong
Description: 
'''
"""
Date: 2022-03-03 09:50:59
LastEditors: yhxiong
LastEditTime: 2022-04-14 15:41:02
Description: 
"""
from yacs.config import CfgNode as CN

_C = CN()



# DATA process related configurations.
_C.DATA = CN()
_C.DATA.CITYFLOW_PATH = "data/AIC22_Track2_NL_Retrieval"
_C.DATA.TRAIN_JSON_PATH = "data/train_tracks_nlpaugv5_shuff_all_cnt.jsonn"
_C.DATA.EVAL_JSON_PATH = "data/train_tracks_nlpaugv5_shuff_all_cnt.json"
_C.DATA.SIZE = 288      # image input dimensions 
_C.DATA.CROP_AREA = 1. ## new_w = CROP_AREA * old_w
_C.DATA.TEST_TRACKS_JSON_PATH = "data/test_tracks.json"
_C.DATA.USE_MOTION = False
_C.DATA.MOTION_PATH = "data/motion_map"
_C.DATA.USE_OTHER_VIEWS = False
_C.DATA.USE_LSTM = False
_C.DATA.USE_NL_MOTION = False
_C.DATA.USE_MAIN_CAR = False
_C.DATA.TEXTS = "single"   # single or concat or seperate
_C.DATA.N_FOLD = 1
_C.DATA.USE_CLIP = False
_C.DATA.CLIP_PATH = "/home/xubocheng/exp/ali/data/clip_map"


# Model specific configurations.
_C.MODEL = CN()

_C.MODEL.NAME = "dual-stream" # base or dual-stream or triple-stream
_C.MODEL.BERT_TYPE = "BERT"
_C.MODEL.BERT_NAME = "bert-base-uncased"
_C.MODEL.IMG_ENCODER = "se_resnext50_32x4d"  # se_resnext50_32x4d, efficientnet-b2, efficientnet-b3
_C.MODEL.LOCAL_IMG_ENCODER = "se_resnext50_32x4d"  # se_resnext50_32x4d, efficientnet-b2, efficientnet-b3
_C.MODEL.GLOBAL_IMG_ENCODER = "se_resnext50_32x4d"  # se_resnext50_32x4d, efficientnet-b2, efficientnet-b3, 3d transformer, video clip
_C.MODEL.RESNET_CHECKPOINT = "checkpoints/motion_effb3_NOCLS_nlpaug_320.pth"
_C.MODEL.NUM_CLASS = 2498
_C.MODEL.EMBED_DIM = 1024   # bert embedding size
_C.MODEL.car_idloss = False
_C.MODEL.mo_idloss = False
_C.MODEL.share_idloss = False
_C.MODEL.box_idloss = False
_C.MODEL.itm_loss = False
_C.MODEL.dropout_vis_backbone = 0.1
_C.MODEL.dropout_vis_backbone_bk = 0.1
_C.MODEL.dropout_vis_backbone_boxes = 0.1

_C.MODEL.TEXTS = "single"   # single or concat or seperate



# Training configurations
_C.TRAIN = CN()
_C.TRAIN.ONE_EPOCH_REPEAT = 30
_C.TRAIN.EPOCH = 40
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_WORKERS = 6
_C.TRAIN.PRINT_FREQ = 40
_C.TRAIN.CONTRASTIVE_LOSS = "infoNCE"  # "infoNCE" or "circle"
_C.TRAIN.CONTINUE = False  # whether to continue training from TEST.RESTORE_FROM
_C.TRAIN.SAVE_FREQ = 2

_C.TRAIN.LR = CN()
_C.TRAIN.LR.BASE_LR = 0.01
_C.TRAIN.LR.WARMUP_EPOCH = 5   # 40
_C.TRAIN.LR.DELAY = 8



# Test configurations
_C.TEST = CN()
_C.TEST.RESTORE_FROM = None
_C.TEST.QUERY_JSON_PATH = "data/test_queries.json"
_C.TEST.BATCH_SIZE = 128
_C.TEST.NUM_WORKERS = 6
_C.TEST.CONTINUE = ""


def get_default_config():
    return _C.clone()