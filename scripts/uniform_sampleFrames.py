import numpy as np
import random


def get_seq_frames(clip_len, num_frames, test_mode=False, keep_head_tail_frames=True):
    """
    Modified from https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159
    Given the video index, return the list of sampled frame indexes.
    Args:
        num_frames (int): Total number of frame in the video.
    Returns:
        seq (list): the indexes of frames of sampled from the video.
    """
    seq = []
    if keep_head_tail_frames:
        seq.append(0)
        clip_len -= 2
        seg_size = float(num_frames - 2) / clip_len
    else:
        seg_size = float(num_frames - 1) / clip_len
    for i in range(clip_len):
        while 1:
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if not test_mode:
                pred = random.randint(start, end)
                if pred not in seq:
                    seq.append(pred)
                    break
            else:
                seq.append((start + end) // 2)
                break
    if keep_head_tail_frames:
        seq.append(num_frames - 1)
    return seq
