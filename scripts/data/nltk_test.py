"""
Date: 2022-03-23 14:12:01
LastEditors: yhxiong
LastEditTime: 2022-03-23 14:21:07
Description: 
"""
import json
from nltk.stem import WordNetLemmatizer
from collections import defaultdict


wnl = WordNetLemmatizer()
actions = [
    "turn",
    "curve",
    "bare",
    "stop",
    "park",
    "wait",
    "pause",
    "merge",
    "switch",
    "change",
    "run",
    "go",
    "drive",
    "keep",
    "cross",
    "move",
    "continue",
    "travel",
    "proceed",
    
    "turns",
    "curves",
    "bares",
    "stops",
    "parks",
    "waits",
    "pauses",
    "merges",
    "switches",
    "changes",
    "runs",
    "goes",
    "drives",
    "keeps",
    "crosses",
    "moves",
    "continues",
    "travels",
    "proceeds",

    "turned",
    "curved",
    "bared",
    "stoped",
    "parked",
    "wait",
    "paused",
    "merged",
    "switched",
    "changed",
    "run",
    "went",
    "drove",
    "kept",
    "crossed",
    "move",
    "continued",
    "traveled",
    "proceeded",

    "turning",
    "curving",
    "bare",
    "stopping",
    "parking",
    "waiting",
    "pause",
    "merging",
    "switching",
    "changing",
    "running",
    "going",
    "driving",
    "keeping",
    "crossing",
    "moving",
    "continue",
    "traveling",
    "proceeding"

]

action_map = defaultdict(list)
for action in actions:
    ori = wnl.lemmatize(action, 'v')
    if action not in action_map[ori]:
        action_map[ori].append(action)

with open("action_map.json", "w") as f:
    json.dump(action_map, f, indent=4)