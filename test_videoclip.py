"""
Date: 2022-04-14 17:00:12
LastEditors: bcxu
LastEditTime: 2022-04-14 17:00:12
Description: Video Encoder in VideoClip as Clip Encoder Test
"""
import json
import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from utils import TqdmToLogger, get_logger, AverageMeter, accuracy, ProgressMeter
from datasets import CityFlowNLDataset
from datasets import VedeoclipInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer, RobertaTokenizer
import pickle
from collections import OrderedDict
from mmpt.models import MMPTModel
import numpy as np
import random

parser = argparse.ArgumentParser(description='AICT5 Training')
parser.add_argument('--config', default="configs/baseline.yaml", type=str,
                    help='config_file')
parser.add_argument('--output_dir', default="output_dir", type=str, help='output dir')
args = parser.parse_args()
out = dict()
use_cuda = True
cfg = get_default_config()
cfg.merge_from_file(args.config)
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

save_dir = os.path.join(cfg.TEST.RESTORE_FROM.split('/')[0], args.output_dir)
os.makedirs(save_dir, exist_ok=True)

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
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
epoch = checkpoint['epoch']
model.load_state_dict(new_state_dict, strict=True)
print(f"Restore from checkpoint: {cfg.TEST.RESTORE_FROM} | epoch: {epoch}")

if use_cuda:
    model.cuda()
    torch.backends.cudnn.benchmark = True

tokenizer, aligner = MMPTModel.from_pretrained("mmpt/how2.yaml", model=False)
with open(cfg.TEST.QUERY_JSON_PATH) as f:
    queries = json.load(f)

test_data = VedeoclipInferenceDataset(cfg.DATA, transform=transform_test)
testloader = DataLoader(dataset=test_data, batch_size=16, shuffle=False)
print(len(testloader))

model.eval()
results = dict()
score_dict = dict()
nlp_num = 4
with torch.no_grad():
    for q_id in tqdm(queries, desc="language"):
        out = {}

        text_clip = queries[q_id]["nl"]
        caps = []
        cmasks = []
        clip_tokens = {}
        for sub_text in text_clip:
            cap, cmask = aligner._build_text_seq(tokenizer(sub_text, add_special_tokens=False)["input_ids"])
            caps.append(cap.numpy())
            cmasks.append(cmask.numpy())
        caps, cmasks = torch.tensor(np.array(caps), dtype=torch.int64), torch.tensor(np.array(cmasks),
                                                                                     dtype=torch.bool)

        for batch_idx, (crop, clip, track_id) in enumerate(testloader):
            bz = crop.shape[0]
            lang_merge_embeds, visual_merge_embeds = model.encode_videoclip(caps.repeat(bz, 1).cuda(),
                                                                            cmasks.repeat(bz, 1).cuda(),
                                                                            crop.repeat_interleave(nlp_num, 0).cuda(),
                                                                            clip.repeat_interleave(nlp_num, 0).cuda())
            for i in range(bz):
                total_score = AverageMeter('MRR', ':6.2f')
                for j in range(nlp_num * i, nlp_num * (i + 1)):
                    total_score.update(
                        float(np.matmul(lang_merge_embeds[j].cpu().numpy(), visual_merge_embeds[j].cpu().numpy().T)))
                out[track_id[i]] = total_score.avg
        score_dict[q_id] = out
        out = sorted(out.items(), key=lambda x: x[1], reverse=True)
        out = [i[0] for i in out]
        results[q_id] = out

save_path = os.path.join(save_dir, 'results_mergefinal.json')
with open(save_path, "w") as f:
    json.dump(results, f, indent=4)

save_path = os.path.join(save_dir, 'score_dict_mergefinal.json')
with open(save_path, "w") as f:
    json.dump(score_dict, f, indent=4)
