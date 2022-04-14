
import imp
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
# from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config import get_default_config
from models.siamese_baseline import SiameseBaselineModelv1,SiameseLocalandMotionModelBIG, SiameseLocalandMotionandLstmModelBIG
from utils import TqdmToLogger, get_logger, AverageMeter, accuracy, ProgressMeter
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer, RobertaTokenizer, CLIPTokenizer, DebertaV2Tokenizer
from collections import OrderedDict
from tensorboardX import SummaryWriter
import datetime
from losses.circle_loss import CircleLoss
from torchmetrics import RetrievalMRR
import itertools
import numpy as np

from mmpt.models import MMPTModel


class WarmUpLR(_LRScheduler):
    def __init__(self, lr_scheduler, warmup_steps, eta_min=1e-7):
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_steps)
                    for base_lr in self.base_lrs]
        return self.lr_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch < self.warmup_steps:
            super().step(epoch)
        else:
            self.last_epoch = epoch
            self.lr_scheduler.step(epoch - self.warmup_steps)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        lr_scheduler = state_dict.pop('lr_scheduler')
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler)

best_top1_eval = 0.
def evaluate(model,valloader,epoch,cfg,index=2):
    global best_top1_eval
    print("Test::::")
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_acc = AverageMeter('Acc@1', ':6.2f')
    top5_acc = AverageMeter('Acc@5', ':6.2f')
    mrrs = AverageMeter('MRR', ':6.2f')
    r1s = AverageMeter('r1', ':6.2f')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, data_time, losses, top1_acc, top5_acc, mrrs, r1s],
        prefix="Test Epoch: [{}]".format(epoch))

    end = time.time()
    with torch.no_grad():
        for batch_idx,batch in enumerate(valloader):

            if cfg.DATA.USE_MOTION:
                if cfg.DATA.USE_LSTM:
                    if cfg.DATA.USE_MAIN_CAR:
                            image, text, text_motion, text_main_car, text_clip, bk, id_car, boxes_points, video_clip = batch
                            main_car_tokens = tokenizer.batch_encode_plus(text_main_car, padding='longest',
                                    return_tensors='pt')
                    else:
                        image, text, text_motion, bk, id_car, boxes_points = batch
                    motion_tokens = tokenizer.batch_encode_plus(text_motion, padding='longest',
                                                        return_tensors='pt')
                else:
                    image, text, bk, id_car = batch
            else:
                if cfg.DATA.USE_LSTM:
                    image, text, text_motion, id_car, boxes_points = batch
                    motion_tokens = tokenizer.batch_encode_plus(text_motion, padding='longest',
                                    return_tensors='pt')

                else:
                    image, text, id_car = batch


            text = [t.split('|')[:3] for t in text]
            tokens = tokenizer.batch_encode_plus(list(itertools.chain(*text)), padding='longest', return_tensors='pt')
            tokens["input_ids"] = tokens["input_ids"].view(len(image), 3, -1)
            tokens["attention_mask"] = tokens["attention_mask"].view(len(image), 3, -1)

            clip_tokens = {"input_ids": "", "attention_mask": ""}
            if cfg.MODEL.GLOBAL_IMG_ENCODER == 'video clip':
                caps = []
                cmasks = []
                clip_tokens = {}
                for sub_text in text_clip:
                    cap, cmask = aligner._build_text_seq(tokenizer(sub_text, add_special_tokens=False)["input_ids"])
                    caps.append(cap.numpy())
                    cmasks.append(cmask.numpy())
                caps, cmasks = torch.tensor(np.array(caps), dtype=torch.int64), torch.tensor(np.array(cmasks), dtype=torch.bool)
                clip_tokens["input_ids"] = caps
                clip_tokens["attention_mask"] = cmasks

            data_time.update(time.time() - end)
            if cfg.DATA.USE_MOTION:
                if cfg.DATA.USE_LSTM:
                    if cfg.DATA.USE_MAIN_CAR:
                        if cfg.DATA.USE_CLIP:

                            pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), 
                                                            motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                                            main_car_tokens['input_ids'].cuda(), main_car_tokens['attention_mask'].cuda(),
                                                            image.cuda(), bk.cuda(), boxes_points.cuda(), clip_tokens['input_ids'].cuda(), clip_tokens['attention_mask'].cuda(), video_clip.cuda())
                        else:
                            pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), 
                                                            motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                                            main_car_tokens['input_ids'].cuda(), main_car_tokens['attention_mask'].cuda(),
                                                            image.cuda(), bk.cuda(), boxes_points.cuda())
                    else:
                        pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                            motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                                            image.cuda(), bk.cuda(), boxes_points.cuda())
                else:
                    pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                        image.cuda(), bk.cuda())
            else:
                if cfg.DATA.USE_LSTM:
                    pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                        motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                                        image.cuda(), boxes_points.cuda())
                else:
                    pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                        image.cuda())
            
            
            logit_scale = logit_scale.mean().exp()
            loss = 0 

            # for visual_embeds,lang_embeds in pairs:
            visual_embeds,lang_embeds = pairs[index]


            sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
            sim_t_2_i = sim_i_2_t.t()
            loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(image.size(0)).cuda())
            loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(image.size(0)).cuda())
            loss += (loss_t_2_i+loss_i_2_t)/2

            mrr = RetrievalMRR()(
            indexes=torch.arange(len(sim_t_2_i))[:, None].expand(len(sim_t_2_i), len(sim_t_2_i)).flatten(),
            preds=sim_t_2_i.flatten(),
            target=torch.eye(len(sim_t_2_i)).cuda().long().bool().flatten()
        )
            
            r = sim_t_2_i.argmax(-1)
            r1 = (r == torch.arange(len(sim_t_2_i)).cuda()).float().mean()
            mrrs.update(mrr, image.size(0))
            r1s.update(r1, image.size(0))

            acc1, acc5 = accuracy(sim_t_2_i, torch.arange(image.size(0)).cuda(), topk=(1, 5))       # 5
            losses.update(loss.item(), image.size(0))
            top1_acc.update(acc1[0], image.size(0))
            top5_acc.update(acc5[0], image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(batch_idx)
    if top1_acc.avg > best_top1_eval:
        best_top1_eval = top1_acc.avg
        checkpoint_file = args.name + "/checkpoint_best_eval.pth"
        torch.save(
            {"epoch": epoch, 
             "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()}, checkpoint_file)
    return top1_acc.avg, top5_acc.avg, losses.avg, mrrs.avg



parser = argparse.ArgumentParser(description='AICT2 Training')
parser.add_argument('--eval', action='store_true', help='resume from checkpoint')
parser.add_argument('--config', default="configs/deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold_videoclip.yaml", type=str,
                    help='config_file')
parser.add_argument('--name', default="deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold_videoclip", type=str,
                    help='experiments')
args = parser.parse_args()

cfg = get_default_config()
cfg.merge_from_file(args.config)

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(cfg.DATA.SIZE, scale=(0.8, 1.)),
    torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(10)], p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])



use_cuda = True

os.makedirs(args.name, exist_ok = True)

if cfg.MODEL.NAME == "base" and not cfg.DATA.USE_MOTION:
    model = SiameseBaselineModelv1(cfg.MODEL)
elif cfg.MODEL.NAME == "dual-stream":
    model = SiameseLocalandMotionModelBIG(cfg.MODEL)

elif cfg.MODEL.NAME == "triple-stream" and cfg.DATA.USE_NL_MOTION:
    model = SiameseLocalandMotionandLstmModelBIG(cfg.MODEL)

else:
    assert cfg.MODEL.NAME in ["base", "dual-stream", "triple-stream"] , "unsupported model"



train_data = CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, transform=transform_test, n_fold=-1, mode='train')
trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=False)
val_data = CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.EVAL_JSON_PATH, transform=transform_test, Random = False, n_fold=-1, mode='val')
valloader = DataLoader(dataset=val_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR.BASE_LR)
step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(trainloader) * cfg.TRAIN.ONE_EPOCH_REPEAT * cfg.TRAIN.LR.DELAY * cfg.DATA.N_FOLD, gamma=0.1)
scheduler = WarmUpLR(lr_scheduler=step_scheduler , warmup_steps=int(1. * cfg.TRAIN.LR.WARMUP_EPOCH*len(trainloader)))

if cfg.MODEL.BERT_TYPE == "BERT":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
elif cfg.MODEL.BERT_TYPE == "ROBERTA":
    tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
elif cfg.MODEL.BERT_TYPE == "CLIP":

    tokenizer = CLIPTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

elif cfg.MODEL.BERT_TYPE == "DEBERTA":
    tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

if cfg.MODEL.GLOBAL_IMG_ENCODER == 'video clip':
    tokenizer, aligner = MMPTModel.from_pretrained("mmpt/how2.yaml", model=False)


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())


epoch = 0
global_step = 0
best_top1 = 0.

if cfg.TRAIN.CONTINUE == True or args.eval:
    new_state_dict = OrderedDict()

    checkpoint = torch.load(cfg.TEST.RESTORE_FROM)
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    global_step = checkpoint['global_step']
    epoch = checkpoint['epoch']
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Restore from checkpoint: {cfg.TEST.RESTORE_FROM} | epoch: {epoch} | global_step: {global_step}")

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # cudnn.benchmark = True


if args.eval:
    evaluate(model,valloader,epoch,cfg,-1)
else:
    writer = SummaryWriter(logdir=os.path.join(os.path.join(args.name, 'logs'), TIMESTAMP))
    for epoch in range(epoch, cfg.TRAIN.EPOCH):

        for fold in range(cfg.DATA.N_FOLD):
            train_data = CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.TRAIN_JSON_PATH, transform=transform_test, n_fold=fold, mode='train')
            trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=False)
            val_data = CityFlowNLDataset(cfg.DATA, json_path=cfg.DATA.EVAL_JSON_PATH, transform=transform_test, Random = False, n_fold=fold, mode='val')
            valloader = DataLoader(dataset=val_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=False)
    
            # model.train()
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            mrrs = AverageMeter('MRR', ':6.2f')
            r1s = AverageMeter('r1', ':6.2f')
            top1_acc = AverageMeter('Acc@1', ':6.2f')
            top5_acc = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(trainloader) * cfg.TRAIN.ONE_EPOCH_REPEAT,
                [batch_time, data_time, losses, top1_acc, top5_acc, mrrs, r1s],
                prefix=f"Epoch [{epoch}] Fold [{fold}] : ")
            end = time.time()
            for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
                for batch_idx, batch in enumerate(trainloader):
                    model.train()
                    if cfg.DATA.USE_MOTION:
                        if cfg.DATA.USE_LSTM:
                            if cfg.DATA.USE_MAIN_CAR:

                                image, text, text_motion, text_main_car, text_clip, bk, id_car, boxes_points, video_clip = batch
                                main_car_tokens = tokenizer.batch_encode_plus(text_main_car, padding='longest',
                                        return_tensors='pt')

                            else:
                                image, text, text_motion, bk, id_car, boxes_points = batch
                            motion_tokens = tokenizer.batch_encode_plus(text_motion, padding='longest',
                                        return_tensors='pt')

                        else:
                            image, text, bk, id_car = batch
                    else:
                        if cfg.DATA.USE_LSTM:
                            image, text, text_motion, id_car, boxes_points = batch
                            motion_tokens = tokenizer.batch_encode_plus(text_motion, padding='longest',
                                        return_tensors='pt')

                        else:
                            image, text, id_car = batch

                    text = [t.split('|')[:3] for t in text]
                    tokens = tokenizer.batch_encode_plus(list(itertools.chain(*text)), padding='longest', return_tensors='pt')
                    tokens["input_ids"] = tokens["input_ids"].view(len(image), 3, -1)
                    tokens["attention_mask"] = tokens["attention_mask"].view(len(image), 3, -1)


                    clip_tokens = {"input_ids": "", "attention_mask": ""}
                    if cfg.MODEL.GLOBAL_IMG_ENCODER == 'video clip':
                        caps = []
                        cmasks = []
                        clip_tokens = {}
                        for sub_text in text_clip:
                            cap, cmask = aligner._build_text_seq(tokenizer(sub_text, add_special_tokens=False)["input_ids"])
                            caps.append(cap.numpy())
                            cmasks.append(cmask.numpy())
                        caps, cmasks = torch.tensor(np.array(caps), dtype=torch.int64), torch.tensor(np.array(cmasks), dtype=torch.bool)
                        clip_tokens["input_ids"] = caps
                        clip_tokens["attention_mask"] = cmasks


                    data_time.update(time.time() - end)
                    global_step += 1
                    optimizer.zero_grad()
                    if cfg.DATA.USE_MOTION:
                        if cfg.DATA.USE_LSTM:
                
                            if cfg.DATA.USE_MAIN_CAR:
                                if cfg.DATA.USE_CLIP:
                                    pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), 
                                                                    motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                                                    main_car_tokens['input_ids'].cuda(), main_car_tokens['attention_mask'].cuda(),
                                                                    image.cuda(), bk.cuda(), boxes_points.cuda(), clip_tokens['input_ids'].cuda(), clip_tokens['attention_mask'].cuda(), video_clip.cuda())
                                else:
                                    pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), 
                                                                    motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                                                    main_car_tokens['input_ids'].cuda(), main_car_tokens['attention_mask'].cuda(),
                                                                    image.cuda(), bk.cuda(), boxes_points.cuda())
                            else:
                                pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(), 
                                                                motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                                                image.cuda(), bk.cuda(), boxes_points.cuda())
                                                                
                        else:
                            pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                                image.cuda(), bk.cuda())
                    else:
                        if cfg.DATA.USE_LSTM:
                            pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                                motion_tokens['input_ids'].cuda(), motion_tokens['attention_mask'].cuda(),
                                                                image.cuda(), boxes_points.cuda())
                        else:
                            pairs, logit_scale, cls_logits = model(tokens['input_ids'].cuda(), tokens['attention_mask'].cuda(),
                                                                image.cuda())
                    logit_scale = logit_scale.mean().exp()
                    loss = 0 
        
                    # symmetric infoNCE / circleloss
                    alpha, n = 0.5, len(pairs)
                    for i, (visual_embeds, lang_embeds) in enumerate(pairs):
                        sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
                        sim_t_2_i = sim_i_2_t.t()
                        if cfg.TRAIN.CONTRASTIVE_LOSS == "infoNCE":
                            loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(image.size(0)).cuda())
                            loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(image.size(0)).cuda())

                        elif cfg.TRAIN.CONTRASTIVE_LOSS == "circle":
                            loss_t_2_i = CircleLoss(similarity='cos')(sim_mat=sim_t_2_i,
                                                                    labels=torch.arange(image.size(0)).cuda())
                            loss_i_2_t = CircleLoss(similarity='cos')(sim_mat=sim_t_2_i,
                                                                    labels=torch.arange(image.size(0)).cuda())
                        loss += ((loss_t_2_i + loss_i_2_t) / 2) * 0.5 ** (n - i)
                    
                    # # instance loss (classification)
                    # for cls_logit in cls_logits:
                    #     loss += 0.5 * F.cross_entropy(cls_logit, id_car.long().cuda())

                    
                    
                    mrr = RetrievalMRR()(
                        indexes=torch.arange(len(sim_t_2_i))[:, None].expand(len(sim_t_2_i), len(sim_t_2_i)).flatten(),
                        preds=sim_t_2_i.flatten(),
                        target=torch.eye(len(sim_t_2_i)).cuda().long().bool().flatten()
                )
                    
                    r = sim_i_2_t.argmax(-1)
                    r1 = (r == torch.arange(len(sim_i_2_t)).cuda()).float().mean()
                    mrrs.update(mrr, image.size(0))
                    r1s.update(r1, image.size(0))


                    acc1, acc5 = accuracy(sim_t_2_i, torch.arange(image.size(0)).cuda(), topk=(1, 5))
                    losses.update(loss.item(), image.size(0))
                    top1_acc.update(acc1[0], image.size(0))
                    top5_acc.update(acc5[0], image.size(0))
                    
                    loss.backward()
                    optimizer.step()
                
                    scheduler.step()
                    batch_time.update(time.time() - end)
                    end = time.time()
                    if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                        progress.display(global_step % (len(trainloader) * cfg.TRAIN.ONE_EPOCH_REPEAT))

                    writer.add_scalar("acc1", acc1[0], global_step + 1)
                    writer.add_scalar("acc5", acc5[0], global_step + 1)
                    writer.add_scalar("mrr", mrr, global_step + 1)
                    writer.add_scalar("r1", r1, global_step + 1)
                    writer.add_scalar("loss", loss.item(), global_step + 1)

                if tmp % 10 == 0:
                    eval_acc1, eval_acc5, eval_loss, eval_mrr = evaluate(model,valloader,epoch,cfg,-1)         
                    writer.add_scalar("mrr_eval", eval_mrr, global_step + 1)
                    writer.add_scalar("acc1_eval", eval_acc1, global_step + 1)
                    writer.add_scalar("acc5_eval", eval_acc5, global_step + 1)
                    writer.add_scalar("loss_eval", eval_loss, global_step + 1)

                    if eval_acc1 > best_top1:
                        best_top1 = eval_acc1
                        checkpoint_file = args.name + "/checkpoint_best.pth"
                        torch.save(
                            {"epoch": epoch, "global_step": global_step,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict()}, checkpoint_file)

            if (epoch+1) % cfg.TRAIN.SAVE_FREQ == 0:
                checkpoint_file = args.name + f"/checkpoint_{epoch}.pth"
                torch.save(
                    {"epoch": epoch, "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, checkpoint_file)


        writer.add_text("top1_acc", str(top1_acc.avg.data))
        writer.add_text("top5_acc", str(top5_acc.avg.data))
