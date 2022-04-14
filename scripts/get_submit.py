"""
Date: 2022-03-03 09:51:12
LastEditors: yhxiong
LastEditTime: 2022-04-14 15:52:45
Description: 
"""
import json
import cv2
import os 
import pickle
import numpy as np
import torch
from numpy import linalg as LA
import torch
import torch.nn.functional as F
from torchmetrics import RetrievalMRR
from tqdm import tqdm 

def get_mean_feats1(img_feat, tacks_ids):
    """对每一帧的特征取平均"""
    mean_gallery = []
    for k in tacks_ids:
        tmp = []
        for fid in img_feat[k]:
            tmp.append(img_feat[k][fid])
        tmp = np.vstack(tmp)
        tmp = np.mean(tmp,0)
        mean_gallery.append(tmp)
    mean_gallery = np.vstack(mean_gallery)
    return mean_gallery

def get_mean_feats1(img_feat, tacks_ids):
    """对每一帧的特征取平均"""
    mean_gallery = []
    for k in tacks_ids:
        tmp = []
        for fid in img_feat[k]:
            tmp.append(img_feat[k][fid])
        tmp = np.vstack(tmp)
        tmp = np.mean(tmp,0)
        mean_gallery.append(tmp)
    mean_gallery = np.vstack(mean_gallery)
    return mean_gallery

def get_all_feats1(img_feat, tacks_ids):
    """取每一帧的特征"""
    mean_gallery = []
    for k in tacks_ids:
        tmp = []
        for fid in  img_feat[k]:
            tmp.append(img_feat[k][fid])
        tmp = np.vstack(tmp)
        # tmp = np.mean(tmp,0)
        mean_gallery.append(tmp)
    mean_gallery = np.vstack(mean_gallery)
    return mean_gallery

def get_mean_feats2(img_feat, tacks_ids):
    mean_gallery = []
    for k in tacks_ids:
       mean_gallery.append(img_feat[(k,)])
    mean_gallery = np.vstack(mean_gallery)
    mean_gallery = torch.from_numpy(mean_gallery)
    mean_gallery = F.normalize(mean_gallery, p = 2, dim = -1).numpy()
    return mean_gallery




used_models1 = ["deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold", "deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold_320", "deberta_allloss_triple_ov_cnt_lang_v5_3sent", "clip_allloss_triple_ov_cnt_all_swin3d", "roberta_allloss"]
used_models2 = ["deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold_videoclip_se_bsz32"]
used_models3 = ["score_dict_mergefinal_2382.json"]

merge_weights1 = [0.8, 0.08, 0.02, 0.1]   

merge_weights2 = [0.05] 


with open("data/test_queries.json") as f:
    queries = json.load(f)
with open("data/test_tracks.json") as f:
    tracks = json.load(f)
query_ids = list(queries.keys())
tacks_ids = list(tracks.keys())
print(len(tacks_ids),len(query_ids))


img_feats1 = []
img_feats2 = []
nlp_feats1 = []
nlp_feats2 = []
score3 = []

for idx, model_name in enumerate(used_models1):
    img_feats1.append(get_mean_feats1(pickle.load(open(f"{model_name}/output/img_feat_{model_name}.pkl",'rb')),tacks_ids))      # 对每一帧的特征取平均
    nlp_feats1.append(pickle.load(open(f"{model_name}/output/lang_feat_{model_name}.pkl",'rb')))

for idx, model_name in enumerate(used_models2):
    img_feats2.append(pickle.load(open(f"{model_name}/output/clip_img_feat_{model_name}.pkl",'rb')))      
    nlp_feats2.append(pickle.load(open(f"{model_name}/output/clip_lang_feat_{model_name}.pkl",'rb')))

for idx, model_name in enumerate(used_models3):
    score3.append(json.load(open(model_name,'r')))      





merge_weights1 = [0.75, 0.1, 0.3, 0.65, 0.25]
merge_weights2 = [0.25]
merge_weights3 = [0.8]

results = dict()

for query in query_ids:
    score = 0.
    for i in range(len(nlp_feats1)):
        q = nlp_feats1[i][query]
        score += merge_weights1[i]*np.mean(np.matmul(q,img_feats1[i].T), 0)


    for i in range(len(nlp_feats2)):
        q = nlp_feats2[i][query][tacks_ids[i]][0]
        img = np.vstack(list(img_feats2[i][query].values()))
        score += merge_weights2[i]*np.matmul(q, img.T)

    for i in range(len(score3)):
        tmp_score = np.array(list(score3[i][query].values()))
        score += merge_weights3[i]*tmp_score

    index = np.argsort(score)[::-1]

    results[query]=[]
    for i in index:
        results[query].append(tacks_ids[i])



with open("submit_results.json", "w") as f:
    json.dump(results, f,indent=4)


