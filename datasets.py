import json
import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from utils import get_logger, create_folds
import pickle
from tqdm import tqdm 
from transformers import BertTokenizer,RobertaTokenizer, CLIPTokenizer, DebertaV2Tokenizer
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, json_path, transform=None, Random=True, max_rnn_length=256, n_fold=0, mode='train'):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = True if data_cfg.TEXTS == "single" else False
        self.max_rnn_length = max_rnn_length
        with open(json_path) as f:
            tracks = json.load(f)

        if self.data_cfg.N_FOLD > 1 and n_fold != -1:
            tracks = create_folds(tracks, num_splits=self.data_cfg.N_FOLD)
            if mode == "train":
                tracks = {track_id: track_data  for track_id, track_data in tracks.items() if track_data["kfold"] != n_fold}
            elif mode == "val":
                tracks = {track_id: track_data  for track_id, track_data in tracks.items() if track_data["kfold"] == n_fold}
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.use_other_views = data_cfg.USE_OTHER_VIEWS
        self.use_nl_motion = data_cfg.USE_NL_MOTION
        self.TEXTS = data_cfg.TEXTS

        self.transform = transform
        self.bk_dic = {}
        self.clip_dic = {}

        self._logger = get_logger()
        
        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.flip_tag = [False] * len(self.list_of_uuids)
        flip_aug = False
        if flip_aug:
            for i in range(len(self.list_of_uuids)):
                text = self.list_of_tracks[i]["nl"]
                for j in range(len(text)):
                    nl = text[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
        
 
        print(len(self.all_indexs))
        print("data load")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
   
        tmp_index = self.all_indexs[index]
        flag = self.flip_tag[index]
        track = self.list_of_tracks[tmp_index]

        # use other views
        nlp_tag = "nl_other_views" if random.random() > 0.5 and self.use_other_views and len(track["nl_other_views"]) > 1 else "nl"


        nl_idx = int(random.uniform(0, len(track[f"{nlp_tag}"])))   
        clip_text = track[f"{nlp_tag}"][nl_idx]

        if self.TEXTS == "cat":
            text = '[SEP]'.join(track[f"{nlp_tag}"][:-1])
            # text = track[f"{nlp_tag}"]
        elif self.TEXTS == "cat2":
            text = ''.join([t.split('.')[1] for t in track[f"{nlp_tag}"][:3]])

        elif self.TEXTS == "cnt":
            cnt_list = [item[1] for item in track["others"]]
            prob = [1/x for x in cnt_list]
            if track["unique"] != []:
                nl_idx = int(random.uniform(0, len(track["unique"])))
                text = track["unique"][nl_idx]
                # 3 sent
                other_texts = text = [data[0] for data in random.choices(track["others"], prob, k=2)]
                text.extend(other_texts)
            else:

                # text = random.choices(track["others"], prob, k=1)[0][0]
                # 3 sent
                text = [data[0] for data in random.choices(track["others"], prob, k=3)]
            # 3 sent
            text = "|".join(text)


        if flag:
            text = text.replace("left","888888").replace("right","left").replace("888888","right")
        

        #### use '(nl)motion' ####
        text_motion = ''
        if self.use_nl_motion:
            nl_idx = int(random.uniform(0, len(track["motion"])))
            text_motion = track["motion"][nl_idx]

        #### use main_car ####
        text_main_car = ''
        if self.use_nl_motion:
            nl_idx = int(random.uniform(0, len(track["main_car"])))
            text_main_car = track["main_car"][nl_idx]

        # add boxes info
        boxes_points = [(box[0], box[1]) for box in track["boxes"]]
        boxes_points = torch.tensor(boxes_points, dtype=torch.float) / 256.0
        if boxes_points.shape[0] >= self.max_rnn_length:  # 最大256batch,大于裁剪，小于填充
            boxes_points = boxes_points[: self.max_rnn_length]
        else:
            boxes_points = torch.cat(
                [
                    boxes_points,
                    torch.zeros(
                        self.max_rnn_length - boxes_points.shape[0],
                        2,
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            )

        # add video clip
        clip = []
        if self.data_cfg.USE_CLIP:
            if self.list_of_uuids[tmp_index] in self.clip_dic:
                clip = self.clip_dic[self.list_of_uuids[tmp_index]]
            else:
                clip_path = open(self.data_cfg.CLIP_PATH + "/%s.pkl" % self.list_of_uuids[tmp_index], 'rb')
                clip = pickle.load(clip_path)
                clip = clip.astype(np.float32)
                clip_path.close()
                self.clip_dic[self.list_of_uuids[tmp_index]] = clip

        frame_idx = int(random.uniform(0, len(track["frames"])))
        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx].replace("./", ""))
        frame = default_loader(frame_path)

        box = track["boxes"][frame_idx]
        if self.crop_area == 1.6666667:
            box = (int(box[0] - box[2] / 3.), int(box[1] - box[3] / 3.), int(box[0] + 4 * box[2] / 3.),
                   int(box[1] + 4 * box[3] / 3.))
        else:
            box = (int(box[0] - (self.crop_area - 1) * box[2] / 2.), int(box[1] - (self.crop_area - 1) * box[3] / 2),
                   int(box[0] + (self.crop_area + 1) * box[2] / 2.), int(box[1] + (self.crop_area + 1) * box[3] / 2.))

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            if self.list_of_uuids[tmp_index] in self.bk_dic:
                bk = self.bk_dic[self.list_of_uuids[tmp_index]]
            else:
                bk = default_loader(self.data_cfg.MOTION_PATH + "/%s.jpg" % self.list_of_uuids[tmp_index])
                self.bk_dic[self.list_of_uuids[tmp_index]] = bk
                bk = self.transform(bk)
                
            if flag:
                crop = torch.flip(crop, [1])
                bk = torch.flip(bk, [1])

            if self.data_cfg.USE_LSTM:
                if self.data_cfg.USE_MAIN_CAR:
                    return crop, text, text_motion, text_main_car, clip_text, bk, tmp_index, boxes_points, clip
                return crop, text, text_motion, bk, tmp_index, boxes_points, clip
            else:
                return crop, text, bk, tmp_index
        if flag:
            crop = torch.flip(crop, [1])
        if self.data_cfg.USE_LSTM:
            if self.data_cfg.USE_CLIP:
                return crop, text, text_motion, tmp_index, boxes_points, clip
            else:
                return crop, text, text_motion, tmp_index, boxes_points

        else:
            return crop, text, tmp_index


class CityFlowNLDatasetFixed(Dataset):
    def __init__(self, data_cfg, json_path, text_embeds, img_embeds, transform=None, Random=True, max_rnn_length=256, n_fold=0, mode='train'):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        with open(json_path) as f:
            tracks = json.load(f)

        if text_embeds is None:
            self.text_embeds = self.save_text_embeds(tracks, text_embeds)
        else:
            with open(text_embeds, "wb") as f:
                self.text_embeds = pickle.load(f)

        # if img_embeds is None:
        #     self.img_embeds = self.save_img_embeds(tracks)
        # else:
        #     with open(img_embeds, "wb") as f:
        #         self.img_embeds = pickle.load(f)


        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = True if data_cfg.TEXTS == "single" else False
        self.max_rnn_length = max_rnn_length


        if self.data_cfg.N_FOLD > 1:
            tracks = create_folds(tracks, num_splits=self.data_cfg.N_FOLD)
            if mode == "train":
                tracks = {track for track in tracks if track["kfold"] != n_fold}
            elif mode == "val":
                tracks = {track for track in tracks if track["kfold"] == n_fold}
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.use_other_views = data_cfg.USE_OTHER_VIEWS
        self.use_nl_motion = data_cfg.USE_NL_MOTION
        self.TEXTS = data_cfg.TEXTS

        self.transform = transform
        self.bk_dic = {}
        self._logger = get_logger()
        
        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.flip_tag = [False] * len(self.list_of_uuids)
        flip_aug = False
        if flip_aug:
            for i in range(len(self.list_of_uuids)):
                text = self.list_of_tracks[i]["nl"]
                for j in range(len(text)):
                    nl = text[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
        
 
        print(len(self.all_indexs))
        print("data load")

    def save_text_embeds(self, tracks, cfg):
        if cfg.MODEL.BERT_TYPE == "BERT":
            model = BertModel.from_pretrained(model_cfg.BERT_NAME)
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif cfg.MODEL.BERT_TYPE == "ROBERTA":
            model = RobertaModel.from_pretrained()
            tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
        elif cfg.MODEL.BERT_TYPE == "CLIP":
            configuration = CLIPTextConfig()
            model = CLIPTextModel(configuration) 
            tokenizer = CLIPTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

        elif cfg.MODEL.BERT_TYPE == "DEBERTA":
            model = DebertaV2Model.from_pretrained(model_cfg.BERT_NAME)
            tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

        text_embeds = {}
        nl_keys = ["nl", "nl_other_views", "motion", "main_car"]
        for track_id, track_data in tqdm(tracks.items()):
            cnt = [data[1] for data in track_data["others"]] if "others" in track_data else []
            for key in nl_keys:
                nl = track_data[key]

                tokens = tokenizer.batch_encode_plus(nl, padding='longest', return_tensors='pt', truncation='longest_first')
                outputs = model(tokens['input_ids'].cuda(), attention_mask=tokens['attention_mask'].cuda())
                # lang_embeds = torch.mean(outputs.last_hidden_state, dim=1)
                track_data[key] = outputs
            nl = track_data["morion"]
            tokens = tokenizer.batch_encode_plus(nl, padding='longest', return_tensors='pt', truncation='longest_first')
            motion_outputs = model(tokens['input_ids'].cuda(), attention_mask=tokens['attention_mask'].cuda())
            # lang_motion_embeds = torch.mean(motion_outputs.last_hidden_state, dim=1)
            track_data["motion"] = motion_outputs

        # with open()

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
   
        tmp_index = self.all_indexs[index]
        flag = self.flip_tag[index]
        track = self.list_of_tracks[tmp_index]

        # use other views
        nlp_tag = "nl_other_views" if random.random() > 0.5 and self.use_other_views and len(track["nl_other_views"]) > 1 else "nl"

        if self.TEXTS == "single":
            nl_idx = int(random.uniform(0, len(track[f"{nlp_tag}"])))
            
            text = track[f"{nlp_tag}"][nl_idx]

        elif self.TEXTS == "cat":
            text = '[SEP]'.join(track[f"{nlp_tag}"][:-1])
            # text = track[f"{nlp_tag}"]
        elif self.TEXTS == "cat2":
            text = ''.join([t.split('.')[1] for t in track[f"{nlp_tag}"][:3]])

        elif self.TEXTS == "cnt":
            cnt_list = [item[1] for item in track["others"]]
            prob = [1/x for x in cnt_list]

            if track["unique"] != []:
                nl_idx = int(random.uniform(0, len(track["unique"])))
                text = track["unique"]
            text = "|".join(text)




        if flag:
            text = text.replace("left","888888").replace("right","left").replace("888888","right")
        
        frame_idx = int(random.uniform(0, len(track["frames"])))
        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx].replace("./", ""))

        
        frame = default_loader(frame_path)

        #### use '(nl)motion' ####
        text_motion = ''
        if self.use_nl_motion:
            if self.random:
                nl_idx = int(random.uniform(0, len(track["motion"])))
            else:
                nl_idx = -1
                # frame_idx = 0
            text_motion = track["motion"][nl_idx]


        # add boxes info
        boxes_points = [(box[0], box[1]) for box in track["boxes"]]
        boxes_points = torch.tensor(boxes_points, dtype=torch.float) / 256.0
        if boxes_points.shape[0] >= self.max_rnn_length:  # 最大256batch,大于裁剪，小于填充
            boxes_points = boxes_points[: self.max_rnn_length]
        else:
            boxes_points = torch.cat(
                [
                    boxes_points,
                    torch.zeros(
                        self.max_rnn_length - boxes_points.shape[0],
                        2,
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            )

        box = track["boxes"][frame_idx]
        if self.crop_area == 1.6666667:
            box = (int(box[0] - box[2] / 3.), int(box[1] - box[3] / 3.), int(box[0] + 4 * box[2] / 3.),
                   int(box[1] + 4 * box[3] / 3.))
        else:
            box = (int(box[0] - (self.crop_area - 1) * box[2] / 2.), int(box[1] - (self.crop_area - 1) * box[3] / 2),
                   int(box[0] + (self.crop_area + 1) * box[2] / 2.), int(box[1] + (self.crop_area + 1) * box[3] / 2.))

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            if self.list_of_uuids[tmp_index] in self.bk_dic:
                bk = self.bk_dic[self.list_of_uuids[tmp_index]]
            else:
                bk = default_loader(self.data_cfg.MOTION_PATH + "/%s.jpg" % self.list_of_uuids[tmp_index])
                self.bk_dic[self.list_of_uuids[tmp_index]] = bk
                bk = self.transform(bk)
                
            if flag:
                crop = torch.flip(crop, [1])
                bk = torch.flip(bk, [1])

            if self.data_cfg.USE_LSTM:
                return crop, text, text_motion, bk, tmp_index, boxes_points
            else:
                return crop, text, bk, tmp_index
        if flag:
            crop = torch.flip(crop, [1])
        if self.data_cfg.USE_LSTM:
            return crop, text, text_motion, tmp_index, boxes_points
        else:
            return crop, text, tmp_index





class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg,transform = None):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        for track_id_index, track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                crop = {"frame": frame_path, "frames_id":frame_idx,"track_id": self.list_of_uuids[track_id_index], "box": box}
                self.list_of_crops.append(crop)
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        track = self.list_of_crops[index]
        frame_path = track["frame"]

        frame = default_loader(frame_path)
        box = track["box"]
        if self.crop_area == 1.6666667:
            box = (int(box[0] - box[2] / 3.), int(box[1] - box[3] / 3.), int(box[0] + 4 * box[2] / 3.),
                   int(box[1] + 4 * box[3] / 3.))
        else:
            box = (int(box[0] - (self.crop_area - 1) * box[2] / 2.), int(box[1] - (self.crop_area - 1) * box[3] / 2),
                   int(box[0] + (self.crop_area + 1) * box[2] / 2.), int(box[1] + (self.crop_area + 1) * box[3] / 2.))

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            bk = default_loader(self.data_cfg.MOTION_PATH + "/%s.jpg" % track["track_id"])
            bk = self.transform(bk)
            return crop, bk, track["track_id"], track["frames_id"]
        return crop, track["track_id"], track["frames_id"]


class CityFlowNLTripleInferenceDataset(Dataset):
    def __init__(self, data_cfg,transform = None, max_rnn_length=256):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        self.max_rnn_length = max_rnn_length

        print(f"Loading tracks from {self.data_cfg.TEST_TRACKS_JSON_PATH}")
        with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        print(len(tracks))
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        for track_id_index, track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                boxes_points = [(box[0], box[1]) for box in track["boxes"]]     
                crop = {"frame": frame_path, "frames_id":frame_idx, "track_id": self.list_of_uuids[track_id_index], "box": box, "boxes_points": boxes_points}
                self.list_of_crops.append(crop)
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        track = self.list_of_crops[index]
        frame_path = track["frame"]

        frame = default_loader(frame_path)

        # add boxes info
        boxes_points = track["boxes_points"]
        boxes_points = torch.tensor(boxes_points, dtype=torch.float) / 256.0
        if boxes_points.shape[0] >= self.max_rnn_length:  # 最大256batch,大于裁剪，小于填充
            boxes_points = boxes_points[: self.max_rnn_length]
        else:
            boxes_points = torch.cat(
                [
                    boxes_points,
                    torch.zeros(
                        self.max_rnn_length - boxes_points.shape[0],
                        2,
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            )


        box = track["box"]
        if self.crop_area == 1.6666667:
            box = (int(box[0] - box[2] / 3.), int(box[1] - box[3] / 3.), int(box[0] + 4 * box[2] / 3.),
                   int(box[1] + 4 * box[3] / 3.))
        else:
            box = (int(box[0] - (self.crop_area - 1) * box[2] / 2.), int(box[1] - (self.crop_area - 1) * box[3] / 2),
                   int(box[0] + (self.crop_area + 1) * box[2] / 2.), int(box[1] + (self.crop_area + 1) * box[3] / 2.))

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            bk = default_loader(self.data_cfg.MOTION_PATH + "/%s.jpg" % track["track_id"])
            bk = self.transform(bk)

            if self.data_cfg.USE_LSTM:
                return crop, bk, boxes_points, track["track_id"], track["frames_id"]
            else:
                return crop, bk, track["track_id"], track["frames_id"]
        if self.data_cfg.USE_LSTM:

            return crop, boxes_points, track["track_id"], track["frames_id"]
        else:
            return crop, track["track_id"], track["frames_id"]


class CityFlowNLQuadraInferenceDataset(Dataset):
    def __init__(self, data_cfg,transform = None, max_rnn_length=256):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        self.max_rnn_length = max_rnn_length

        print(f"Loading tracks from {self.data_cfg.TEST_TRACKS_JSON_PATH}")
        with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        print(len(tracks))
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.clip_dic = {}
        self.bk_dic = {}
        for track_id_index, track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                boxes_points = [(box[0], box[1]) for box in track["boxes"]]     
                crop = {"frame": frame_path, "frames_id":frame_idx, "track_id": self.list_of_uuids[track_id_index], "box": box, "boxes_points": boxes_points}
                self.list_of_crops.append(crop)
        self._logger = get_logger()

    def __len__(self):
        return len(self.all_indexs)
        # return len(self.list_of_crops)

    def __getitem__(self, index):
        tmp_index = self.all_indexs[index]
        track = self.list_of_tracks[tmp_index]
        frame_idx = int(random.uniform(0, len(track["frames"])))
        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx].replace("./", ""))
        frame = default_loader(frame_path)

        # track = self.list_of_crops[index]
        # frame_path = track["frame"]
        # track_id = track["track_id"]
        # frame = default_loader(frame_path)


        # add boxes info
        # boxes_points = track["boxes_points"]
        boxes_points = [(box[0], box[1]) for box in track["boxes"]]  
        boxes_points = torch.tensor(boxes_points, dtype=torch.float) / 256.0
        if boxes_points.shape[0] >= self.max_rnn_length:  # 最大256batch,大于裁剪，小于填充
            boxes_points = boxes_points[: self.max_rnn_length]
        else:
            boxes_points = torch.cat(
                [
                    boxes_points,
                    torch.zeros(
                        self.max_rnn_length - boxes_points.shape[0],
                        2,
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            )

        # add video clip
        # if self.data_cfg.USE_CLIP:
        #     if track_id in self.clip_dic:
        #         clip = self.clip_dic[track_id]
        #     else:
        #         clip_path = open(self.data_cfg.CLIP_PATH + "/%s.pkl" % track_id, 'rb')
        #         clip = pickle.load(clip_path)
        #         clip = clip.astype(np.float32)
        #         clip_path.close()
        #         self.clip_dic[track_id] = clip

        if self.data_cfg.USE_CLIP:
            if self.list_of_uuids[tmp_index] in self.clip_dic:
                clip = self.clip_dic[self.list_of_uuids[tmp_index]]
            else:
                clip_path = open(self.data_cfg.CLIP_PATH + "/%s.pkl" % self.list_of_uuids[tmp_index], 'rb')
                clip = pickle.load(clip_path)
                clip = clip.astype(np.float32)
                clip_path.close()
                self.clip_dic[self.list_of_uuids[tmp_index]] = clip


        # box = track["box"]
        box = track["boxes"][frame_idx]

        if self.crop_area == 1.6666667:
            box = (int(box[0] - box[2] / 3.), int(box[1] - box[3] / 3.), int(box[0] + 4 * box[2] / 3.),
                   int(box[1] + 4 * box[3] / 3.))
        else:
            box = (int(box[0] - (self.crop_area - 1) * box[2] / 2.), int(box[1] - (self.crop_area - 1) * box[3] / 2),
                   int(box[0] + (self.crop_area + 1) * box[2] / 2.), int(box[1] + (self.crop_area + 1) * box[3] / 2.))

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:

            # bk = default_loader(self.data_cfg.MOTION_PATH + "/%s.jpg" % track["track_id"])
            # bk = self.transform(bk)

            if self.list_of_uuids[tmp_index] in self.bk_dic:
                bk = self.bk_dic[self.list_of_uuids[tmp_index]]
            else:
                bk = default_loader(self.data_cfg.MOTION_PATH + "/%s.jpg" % self.list_of_uuids[tmp_index])
                self.bk_dic[self.list_of_uuids[tmp_index]] = bk
                bk = self.transform(bk)

            if self.data_cfg.USE_LSTM:
                # return crop, bk, boxes_points, clip, track["track_id"], track["frames_id"]
                return crop, bk, boxes_points, clip, self.list_of_uuids[tmp_index], frame_idx
            else:
                return crop, bk, track["track_id"], track["frames_id"]
        if self.data_cfg.USE_LSTM:

            return crop, boxes_points, track["track_id"], track["frames_id"]
        else:
            return crop, track["track_id"], track["frames_id"]

class CityFlowNLQuadraInferenceImgDataset(Dataset):
    def __init__(self, data_cfg,transform = None, max_rnn_length=256):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        self.max_rnn_length = max_rnn_length

        print(f"Loading tracks from {self.data_cfg.TEST_TRACKS_JSON_PATH}")
        with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        print(len(tracks))
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.clip_dic = {}
        self.bk_dic = {}
        for track_id_index, track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                boxes_points = [(box[0], box[1]) for box in track["boxes"]]     
                crop = {"frame": frame_path, "frames_id":frame_idx, "track_id": self.list_of_uuids[track_id_index], "box": box, "boxes_points": boxes_points}
                self.list_of_crops.append(crop)
        self._logger = get_logger()

    def __len__(self):
        # return len(self.all_indexs)
        return len(self.list_of_crops)

    def __getitem__(self, index):
        # tmp_index = self.all_indexs[index]
        # track = self.list_of_tracks[tmp_index]
        # frame_idx = int(random.uniform(0, len(track["frames"])))
        # frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx].replace("./", ""))
        # frame = default_loader(frame_path)

        track = self.list_of_crops[index]
        frame_path = track["frame"]
        track_id = track["track_id"]
        frame = default_loader(frame_path)


        # add boxes info
        boxes_points = track["boxes_points"]
        # boxes_points = [(box[0], box[1]) for box in track["boxes"]]  
        boxes_points = torch.tensor(boxes_points, dtype=torch.float) / 256.0
        if boxes_points.shape[0] >= self.max_rnn_length:  # 最大256batch,大于裁剪，小于填充
            boxes_points = boxes_points[: self.max_rnn_length]
        else:
            boxes_points = torch.cat(
                [
                    boxes_points,
                    torch.zeros(
                        self.max_rnn_length - boxes_points.shape[0],
                        2,
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            )

        # add video clip
        if self.data_cfg.USE_CLIP:
            if track_id in self.clip_dic:
                clip = self.clip_dic[track_id]
            else:
                clip_path = open(self.data_cfg.CLIP_PATH + "/%s.pkl" % track_id, 'rb')
                clip = pickle.load(clip_path)
                clip = clip.astype(np.float32)
                clip_path.close()
                self.clip_dic[track_id] = clip

        # if self.data_cfg.USE_CLIP:
        #     if self.list_of_uuids[tmp_index] in self.clip_dic:
        #         clip = self.clip_dic[self.list_of_uuids[tmp_index]]
        #     else:
        #         clip_path = open(self.data_cfg.CLIP_PATH + "/%s.pkl" % self.list_of_uuids[tmp_index], 'rb')
        #         clip = pickle.load(clip_path)
        #         clip = clip.astype(np.float32)
        #         clip_path.close()
        #         self.clip_dic[self.list_of_uuids[tmp_index]] = clip


        box = track["box"]
        # box = track["boxes"][frame_idx]

        if self.crop_area == 1.6666667:
            box = (int(box[0] - box[2] / 3.), int(box[1] - box[3] / 3.), int(box[0] + 4 * box[2] / 3.),
                   int(box[1] + 4 * box[3] / 3.))
        else:
            box = (int(box[0] - (self.crop_area - 1) * box[2] / 2.), int(box[1] - (self.crop_area - 1) * box[3] / 2),
                   int(box[0] + (self.crop_area + 1) * box[2] / 2.), int(box[1] + (self.crop_area + 1) * box[3] / 2.))

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:

            bk = default_loader(self.data_cfg.MOTION_PATH + "/%s.jpg" % track["track_id"])
            bk = self.transform(bk)

            # if self.list_of_uuids[tmp_index] in self.bk_dic:
            #     bk = self.bk_dic[self.list_of_uuids[tmp_index]]
            # else:
            #     bk = default_loader(self.data_cfg.MOTION_PATH + "/%s.jpg" % self.list_of_uuids[tmp_index])
            #     self.bk_dic[self.list_of_uuids[tmp_index]] = bk
            #     bk = self.transform(bk)

            if self.data_cfg.USE_LSTM:
                return crop, bk, boxes_points, clip, track["track_id"], track["frames_id"]
                # return crop, bk, boxes_points, clip, self.list_of_uuids[tmp_index], frame_idx
            else:
                return crop, bk, track["track_id"], track["frames_id"]
        if self.data_cfg.USE_LSTM:

            return crop, boxes_points, track["track_id"], track["frames_id"]
        else:
            return crop, track["track_id"], track["frames_id"]



