'''
Author: Catherine Xiong
Date: 2022-03-22 17:06:50
LastEditTime: 2022-04-09 11:18:25
LastEditors: Catherine Xiong
Description: 
'''
"""
Date: 2022-03-22 17:06:50
LastEditors: yhxiong
LastEditTime: 2022-04-08 16:35:28
Description: 
"""
import json
import sys
from torchmetrics import RetrievalMRR

def main(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    score = 0
    cnt = 0
    acc1 = 0
    length = len(data)
    for true_id, res in data.items():
        pred_idx = res.index(true_id)
        print(cnt, pred_idx)
        score += 1 / (1 + pred_idx)
        cnt += 1
        if pred_idx == 0: acc1 += 1
        if cnt == length:
            break
    mrr = score / cnt
    # mrr = score / len(data)
    print(f"MRR: {mrr}")

    acc1 = acc1 / cnt
    print(f"ACC1: {acc1}")



if __name__ == '__main__':
    main("/home/xiongyihua/comp/ali/deberta_allloss_triple_ov_cnt_lang_v5_3sent_10fold_videoclip_se_bsz32_val.json")   
    # main(sys.argv[1:])


