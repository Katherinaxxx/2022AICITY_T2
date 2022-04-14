"""
Date: 2022-03-17 15:37:37
LastEditors: yhxiong
LastEditTime: 2022-03-22 15:20:33
Description: 
"""
import os
import cv2
import argparse
import json


def combine_frms(frames_list, video_id, save_dir):
    img_array=[]
    for filename in frames_list:
        img = cv2.imread(filename.replace('./', '/home/xiongyihua/comp/data/AIC22_Track2_NL_Retrieval/'))
        img_array.append(img)
    w, h, c = img_array[0].shape
    # video_path = os.path.join(save_dir, str(video_id) + ".avi")
    # fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    video_path = os.path.join(save_dir, str(video_id) + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fourcc = cv2.VideoWriter_fourcc(*'AVC1')  # not work on this machine
    videowrite = cv2.VideoWriter(video_path, fourcc,20, (h, w), True)

    for img in img_array:
        videowrite.write(img)
    videowrite.release() 
    print(f"Save to {video_path}.")

def main(args):

    with open(args.track_data, "r") as f:
        data = json.load(f)
    
    order_json = {}
    for idx, (track_id, track_data) in enumerate(data.items()):
        combine_frms(track_data["frames"], idx, args.save_dir)
        order_json[track_id] = idx
    
    with open("oder.json", "w") as f:
        json.dump(order_json, f,indent=4)


if __name__ == '__main__':
    print("Loading parameters...")
    parser = argparse.ArgumentParser(description='Combine frames into video')
    parser.add_argument('--track_data', dest='track_data', default='/home/xiongyihua/comp/data/train_tracks.json',
                        help='dataset root path')
    parser.add_argument('--save_dir', dest='save_dir', default='/home/xiongyihua/comp/ali/visualize_tool/video_mp4_train',
                        help='dataset root path')

    args = parser.parse_args()

    main(args)