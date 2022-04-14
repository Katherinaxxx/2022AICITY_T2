"""
Date: 2022-03-03 09:51:12
LastEditors: yhxiong
LastEditTime: 2022-04-14 15:39:03
Description: 
"""
import json
import math
import os
import sys
from datetime import datetime
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config import get_default_config
from models.siamese_baseline import SiameseBaselineModelv1,SiameseLocalandMotionModelBIG, SiameseLocalandMotionandLstmModelBIG
from utils import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset, CityFlowNLTripleInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer, CLIPTokenizer, DebertaV2Tokenizer
import pickle
from collections import OrderedDict

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

parser = argparse.ArgumentParser(description='AICT5 Training')
parser.add_argument('--config', default="configs/deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold_320.yaml", type=str,
                    help='config_file')
args = parser.parse_args()
out = dict()


use_cuda = True
cfg = get_default_config()
cfg.merge_from_file(args.config)
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
save_dir = cfg.TEST.RESTORE_FROM.split('/')[0] + "/output"
os.makedirs(save_dir,exist_ok = True)
save_name = args.config.split('/')[-1].split('.')[0]


if cfg.MODEL.NAME == "base" and not cfg.DATA.USE_MOTION:
    model = SiameseBaselineModelv1(cfg.MODEL)
elif cfg.MODEL.NAME == "dual-stream":
    model = SiameseLocalandMotionModelBIG(cfg.MODEL)

elif cfg.MODEL.NAME == "triple-stream" and cfg.DATA.USE_NL_MOTION:
    model = SiameseLocalandMotionandLstmModelBIG(cfg.MODEL)

else:
    assert cfg.MODEL.NAME in ["base", "dual-stream", "triple-stream"] , "unsupported model"
checkpoint = torch.load(cfg.TEST.RESTORE_FROM)

new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
global_step = checkpoint['global_step']
epoch = checkpoint['epoch']
model.load_state_dict(new_state_dict,strict=False)
print(f"Restore from checkpoint: {cfg.TEST.RESTORE_FROM} | epoch: {epoch} | global_step: {global_step}")

if use_cuda:
    model.cuda()
    torch.backends.cudnn.benchmark = True

if cfg.MODEL.NAME == "triple-stream" and cfg.DATA.USE_NL_MOTION:
    test_data=CityFlowNLTripleInferenceDataset(cfg.DATA, transform=transform_test)
else:
    test_data=CityFlowNLInferenceDataset(cfg.DATA, transform=transform_test)
testloader = DataLoader(dataset=test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=8)

if cfg.MODEL.BERT_TYPE == "BERT":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
elif cfg.MODEL.BERT_TYPE == "ROBERTA":
    tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
elif cfg.MODEL.BERT_TYPE == "CLIP":
    tokenizer = CLIPTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
elif cfg.MODEL.BERT_TYPE == "DEBERTA":
    tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

model.eval()

with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(testloader), desc="vision"):

        if cfg.MODEL.NAME == "dual-stream":
            image, motion, track_id, frames_id = batch

            vis_embed = model.encode_images(image.cuda(), motion.cuda())
        else:
            image, motion, boxes_points, track_id, frames_id = batch

            vis_embed = model.encode_images(image.cuda(), motion.cuda(), boxes_points.cuda())
        for  i in range(len(track_id)):
            if track_id[i] not in out:
                out[track_id[i]]=dict()
            out[track_id[i]][frames_id[i].item()] = vis_embed[i,:].data.cpu().numpy()

pickle.dump(out, open(save_dir+'/img_feat_%s.pkl' % save_name, 'wb'))

with open(cfg.TEST.QUERY_JSON_PATH) as f:
    queries = json.load(f)

query_embed = dict()
import random
with torch.no_grad():
    for q_id in tqdm(queries, desc="language"):
        #### use '(nl)motion' ####
        text_motion = ''
        nl_idx = int(random.uniform(0, len(queries[q_id]["motion"])))
        text_motion = queries[q_id]["motion"][nl_idx]

        #### use main_car ####
        text_main_car = ''
        nl_idx = int(random.uniform(0, len(queries[q_id]["main_car"])))
        text_main_car = queries[q_id]["main_car"][nl_idx]

        main_car_tokens = tokenizer.batch_encode_plus([text_main_car], padding='longest', return_tensors='pt')
        motion_tokens = tokenizer.batch_encode_plus([text_motion], padding='longest', return_tensors='pt')
        tokens = tokenizer.batch_encode_plus(queries[q_id]["nl"][:3], padding='longest', return_tensors='pt')

        if cfg.MODEL.NAME == "dual-stream":
            lang_embeds = model.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), main_car_tokens['input_ids'].cuda(), main_car_tokens['attention_mask'].cuda())
        else:
            lang_embeds = model.encode_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(), main_car_tokens['input_ids'].cuda(), main_car_tokens['attention_mask'].cuda())
            
        query_embed[q_id] = lang_embeds.data.cpu().numpy()
pickle.dump(query_embed,open(save_dir+'/lang_feat_%s.pkl'%save_name, 'wb'))
torch.cuda.empty_cache()