"""
Date: 2022-03-22 10:20:31
LastEditors: yhxiong
LastEditTime: 2022-03-28 16:55:53
Description: 
"""
import json
import sys
from tqdm import tqdm
import itertools


def add_cnt_fn(data, cnt_data):
    for track_id, track_data in tqdm(data.items()):
        unique = []
        others = []
        for nl in list(itertools.chain(*[track_data["nl"][:3], track_data["nl_other_views"]])):
            print(nl, nl.split("."))
            # cnt = cnt_data['.'.join(nl.split(".")[1:]).strip()]
            cnt = cnt_data[nl.strip()]
            if cnt == 1:
                unique.append(nl)
            else:
                others.append([nl, cnt])
        track_data["unique"] = unique
        track_data["others"] = others
    return data

 
if __name__ == '__main__':
    # python scripts/add_cnt.py ../data/test_queries_nlpaugv5.json ../data/test_tracks_cnt.json
    # python scripts/add_cnt.py ../data/train_tracks_nlpaugv5_shuff_train.json ../data/train_tracks_cnt.json
    # python scripts/add_cnt.py ../data/train_tracks_nlpaugv5_shuff_val.json ../data/train_tracks_cnt.json
    # python scripts/add_cnt.py ../data/train_tracks_nlpaugv5_shuff_all.json ../data/train_tracks_cnt.json
	with open(sys.argv[1]) as f:
		data = json.load(f)
	with open(sys.argv[2]) as f:
		cnt_data = json.load(f)

	add_cnt_data = add_cnt_fn(data, cnt_data)	# v3

	with open(sys.argv[1].replace(".json", "_cnt.json"), "w") as f:
		json.dump(add_cnt_data, f,indent=4)

