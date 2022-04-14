import cv2
import json
import numpy as np
from tqdm import tqdm
import os
import multiprocessing
from functools import partial
import pickle
from uniform_sampleFrames import get_seq_frames

imgpath = "../data/AIC22_Track2_NL_Retrieval/"
with open("../data/test_tracks.json") as f:
    tracks_test = json.load(f)
with open("../data/train_tracks.json") as f:
    tracks_train = json.load(f)
all_tracks = tracks_train
# 整合tracks_train和tracks_test
for track in tracks_test:
    all_tracks[track] = tracks_test[track]

n_worker = 12
import glob

save_clip_dir = "../data/clip_map"
os.makedirs(save_clip_dir, exist_ok=True)


def get_clip_map(info):
    track, track_id = info
    frames = []
    for i in range(len(track["frames"])):
        frame_path = track["frames"][i]
        frame_path = os.path.join(imgpath, frame_path)
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    # TODO Sample帧
    if len(frames) > 32:
        frame_sample = get_seq_frames(32, len(frames))
        # print(len(frames), frame_sample)
        frames = [frames[frame_index] for frame_index in frame_sample]
    frames_arr = np.array(frames)
    if frames_arr.shape[0] < 32:
        frames_arr = np.concatenate((np.zeros((32-frames_arr.shape[0], 224, 224, 3)), frames_arr), axis=0)
    assert frames_arr.shape == (32, 224, 224, 3), "unsupported frames_arr"
    frames_arr = np.transpose(frames_arr, (3, 0, 1, 2))
    # frames_arr = frames_arr[np.newaxis, :]
    output = open(save_clip_dir + "/%s.pkl" % track_id, 'wb')
    pickle.dump(frames_arr, output)
    output.close()


files = []
for track_id in all_tracks:
    files.append((all_tracks[track_id], track_id))

with multiprocessing.Pool(n_worker) as pool:
    for imgs in tqdm(pool.imap_unordered(get_clip_map, files)):
        pass
