"""
Date: 2022-04-06 16:55:55
LastEditors: yhxiong
LastEditTime: 2022-04-14 15:37:23
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
from models.siamese_baseline import SiameseBaselineModelv1, SiameseLocalandMotionModelBIG, \
    SiameseLocalandMotionandLstmModelBIG
from utils import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter
from datasets import CityFlowNLQuadraInferenceDataset, CityFlowNLQuadraInferenceImgDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer
import pickle
from collections import OrderedDict
from mmpt.models import MMPTModel
import numpy as np

parser = argparse.ArgumentParser(description='AICT5 Training')
parser.add_argument('--config', default="configs/deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold_videoclip_se_bsz32.yaml", type=str,
                    help='config_file')
args = parser.parse_args()
use_cuda = True
cfg = get_default_config()
cfg.merge_from_file(args.config)
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
save_dir = cfg.TEST.RESTORE_FROM.split('/')[0] + "/output2"
os.makedirs(save_dir,exist_ok = True)
save_name = args.config.split('/')[-1].split('.')[0]
if cfg.MODEL.NAME == "base":
    model = SiameseBaselineModelv1(cfg.MODEL)
elif cfg.MODEL.NAME == "dual-stream":
    model = SiameseLocalandMotionModelBIG(cfg.MODEL)
elif cfg.MODEL.NAME == "triple-stream":
    model = SiameseLocalandMotionandLstmModelBIG(cfg.MODEL)
else:
    assert cfg.MODEL.NAME in ["base", "dual-stream", "triple-stream"], "unsupported model"

checkpoint = torch.load(cfg.TEST.RESTORE_FROM)
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
epoch = checkpoint['epoch']
model.load_state_dict(new_state_dict,strict=True)
print(f"Restore from checkpoint: {cfg.TEST.RESTORE_FROM} | epoch: {epoch}")

if use_cuda:
    model.cuda()
    torch.backends.cudnn.benchmark = True

tokenizer, aligner = MMPTModel.from_pretrained("mmpt/how2.yaml", model=False)
with open(cfg.TEST.QUERY_JSON_PATH) as f:
    queries = json.load(f)


model.eval()

### image ####
out = dict()
test_data = CityFlowNLQuadraInferenceImgDataset(cfg.DATA, transform=transform_test)
testloader = DataLoader(dataset=test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)
with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(testloader), desc="vision"):
        image, motion, boxes_points, clip, track_id, frames_id = batch

        vis_embed = model.encode_clip_images(image.cuda(), motion.cuda(), boxes_points.cuda())
        for  i in range(len(track_id)):
            if track_id[i] not in out:
                out[track_id[i]]=dict()
            out[track_id[i]][frames_id[i].item()] = vis_embed[i,:].data.cpu().numpy()

pickle.dump(out, open(save_dir+'/img_feat_%s.pkl' % save_name, 'wb'))


### nl ####
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

        lang_embeds = model.encode_clip_text(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(), main_car_tokens['input_ids'].cuda(), main_car_tokens['attention_mask'].cuda())
            
        query_embed[q_id] = lang_embeds.data.cpu().numpy()
pickle.dump(query_embed,open(save_dir+'/lang_feat_%s.pkl'%save_name, 'wb'))


## video clip ####
videoclip_out = dict()

test_data = CityFlowNLQuadraInferenceDataset(cfg.DATA, transform=transform_test)
testloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

with torch.no_grad():
    for batch_idx, (image, motion, boxes_points, clip, track_id, frames_id) in tqdm(enumerate(testloader), desc="video clip"):
        videoclip_out[track_id] = {}

        for q_id in queries:
        
            img_feat, nl_feat = [], []
            for sub_nl in queries[q_id]['nl']:
                caps = []
                cmasks = []
                cap, cmask = aligner._build_text_seq(
                    tokenizer(sub_nl, add_special_tokens=False)["input_ids"])

                caps.append(cap.numpy())
                cmasks.append(cmask.numpy())
                caps, cmasks = torch.tensor(np.array(caps), dtype=torch.int64), torch.tensor(np.array(cmasks), dtype=torch.bool)

                lang_merge_embeds, visual_merge_embeds = model.encode_videoclip(caps.cuda(), cmasks.cuda(), clip.cuda())
                img_feat.append(visual_merge_embeds.data.cpu().numpy())
                nl_feat.append(lang_merge_embeds.data.cpu().numpy())
            img_feat = np.mean(np.vstack(img_feat), 0)
            nl_feat = np.mean(np.vstack(nl_feat), 0)

            videoclip_out[track_id][q_id] = [img_feat, nl_feat]


pickle.dump(videoclip_out, open(save_dir+'/videoclip_feat_%s.pkl' % save_name, 'wb'))


### merge ####
videoclip = pickle.load(open(save_dir+'/videoclip_feat_%s.pkl' % save_name,'rb'))
lang_embeds = pickle.load(open(save_dir+'/lang_feat_%s.pkl' % save_name,'rb'))
img_embeds = pickle.load(open(save_dir+'/img_feat_%s.pkl' % save_name,'rb'))

lang_dict = dict()
img_dict = dict()
cnt = 0
for track_id, feats in tqdm(videoclip.items()):
    for q_id, feat in feats.items():
        img_feat, nl_feat = feat
        if q_id not in lang_dict:
            lang_dict[q_id] = {}
        if q_id not in img_dict:
            img_dict[q_id] = {}
        lang_dict[q_id][track_id[0]] = model.merge_text(torch.from_numpy(nl_feat).unsqueeze(0).cuda(), torch.from_numpy(lang_embeds[q_id]).cuda()).data.cpu().numpy()
        
        tmp = []
        for frame_id, img_embed in img_embeds[track_id[0]].items():
            tmp.append(model.merge_image(torch.from_numpy(img_feat).unsqueeze(0).cuda(), torch.from_numpy(img_embed).unsqueeze(0).cuda()).data.cpu().numpy())
            break
        tmp = np.vstack(tmp)
        tmp = np.mean(tmp,0)
        img_dict[q_id][track_id[0]] = tmp

        # img_embed = np.vstack(list(img_embeds[track_id[0]].values()))
        # img_embed = np.mean(img_embed,0)
        # img_dict[q_id][track_id[0]] = model.merge_image(torch.from_numpy(img_feat).unsqueeze(0).cuda(), torch.from_numpy(img_embed).unsqueeze(0).cuda()).data.cpu().numpy()
                
                
pickle.dump(img_dict, open(save_dir+'/clip_img_feat_%s.pkl' % save_name, 'wb'))
pickle.dump(lang_dict, open(save_dir+'/clip_lang_feat_%s.pkl'%save_name, 'wb'))


#### result ####
lang_dict = pickle.load(open(save_dir+'/clip_lang_feat_%s.pkl' % save_name,'rb'))
img_dict = pickle.load(open(save_dir+'/clip_img_feat_%s.pkl' % save_name,'rb'))
results = dict()


# tacks_ids = lang_dict[q_id].keys()
for q_id in lang_dict:
    tacks_ids = list(lang_dict[q_id].keys())

    score = []
    for track_id, img in img_dict[q_id].items():
        q = lang_dict[q_id][track_id]
        img = img_dict[q_id][track_id]
        score.append(np.matmul(q, img.T)[0])



    index = np.argsort(score)[::-1]

    results[q_id] = []
    for i in index:
        results[q_id].append(tacks_ids[i])

with open("deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold_videoclip_se_bsz32_val2.json", "w") as f:
    json.dump(results, f,indent=4)


