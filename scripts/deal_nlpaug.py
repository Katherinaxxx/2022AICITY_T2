"""
Date: 2022-03-03 09:51:12
LastEditors: yhxiong
LastEditTime: 2022-04-14 15:46:59
Description: nouns
Extract the nouns of each sentence and put them at the beginning of the sentence. 
And combine all nouns into a new sentence
"""
import json
import sys
import spacy
from tqdm import tqdm
from flashtext import KeywordProcessor
import random
import collections

nlp = spacy.load("en_core_web_sm")


USE_OTHER_VIREWS = True


def load_map(json_path):
	with open(json_path, 'r') as f:
		ori_map = json.load(f)
	new_map = collections.defaultdict(str)
	all_v_list = []
	for k, v_list in ori_map.items():
		all_v_list.extend(v_list)
		for v in v_list:
			new_map[v] = k
	return new_map, all_v_list


def add_clean(train):
	action_map, actions_list = load_map("scripts/data/action_map.json")
	car_map, cars_list = load_map("scripts/data/vehicle_group_v1_rep.json")
	color_map, colors_list = load_map("scripts/data/color_group_v1.json")
	action_keyword_processor = load_keyword_processor(actions_list, False)
	color_keyword_processor = load_keyword_processor(colors_list, False)
	vehicle_keyword_processor = load_keyword_processor(cars_list, False)

	cnt = 0
	track_ids = list(train.keys())
	for id_ in tqdm(track_ids):
		main_car = []
		other_cars = []
		train[id_]["motion"] = []
		for i, text in enumerate(train[id_]["nl"]):

			actions = [action_map[action] for action in action_keyword_processor.extract_keywords(text)]
			colors = [color_map[color] for color in color_keyword_processor.extract_keywords(text)]
			vehicles = [car_map[car] for car in vehicle_keyword_processor.extract_keywords(text)]

			if colors == []:
				colors = ['']
			if vehicles == []:
				vehicles = ['car']
			if actions == []:
				actions = ['']
			main_car.append(' '.join([colors[0], vehicles[0]]))

			if len(actions) == len(vehicles) == 2:

				cnt += 1
				if len(colors) == 1:
					colors.append("")
				other_cars.append(' '.join([colors[1], vehicles[1], actions[1]])+'.')
			


			# motion
			root = ''
			motion_tokens = []
			key_action = action_keyword_processor.extract_keywords(text)
			key_action = [action for action in key_action if action_map[action] not in ["behind", "follow"]]
			if key_action == []:
				continue


			doc = nlp(text)
			for token in doc:
				if token.text in key_action:
					root = token.text
					motion_tokens.append(action_map[root])
				ancestors = list(token.ancestors)
				if root is not '' and root in [str(ancestors[i]) for i in range(len(ancestors))]:
					motion_tokens.append(token.text)
				if root is not '' and token.tag_ in ["NN", "NNS", "RB", "JJ"]:
					root = ''
			if len(motion_tokens) < 2:
				continue
			motion = ' '.join(motion_tokens) + '.'
			train[id_]["motion"].append(motion)
			if train[id_]["motion"] == []:
				print(text)
				break
		train[id_]["main_car"] = main_car
		train[id_]["other_cars"] = other_cars

	
	if USE_OTHER_VIREWS:
		for id_ in tqdm(track_ids):
			main_car = []
			other_cars = []
			for i, text in enumerate(train[id_]["nl_other_views"]):
				# find kw and convert to unif

				actions = [action_map[action] for action in action_keyword_processor.extract_keywords(text)]
				colors = [color_map[color] for color in color_keyword_processor.extract_keywords(text)]
				vehicles = [car_map[car] for car in vehicle_keyword_processor.extract_keywords(text)]
				if actions == [""] or colors == [""] or vehicles == [""]:
					print(text)
				if colors == []:
					colors = ['']
				if vehicles == []:
					vehicles = ['car']
				if actions == []:
					actions = ['']
				main_car.append(' '.join([colors[0], vehicles[0]]))
				if len(vehicles) == len(actions) == len(colors) == 2:
					cnt += 1
					other_cars.append(' '.join([colors[1], vehicles[1], actions[1]])+'.')



				# motion
				root = ''
				motion_tokens = []
				key_action = action_keyword_processor.extract_keywords(text)
				key_action = [action for action in key_action if action_map[action] not in ["behind", "follow"]]
				if key_action == []:
					continue


				doc = nlp(text)
				for token in doc:
					if token.text in key_action:
						root = token.text
						motion_tokens.append(action_map[root])
					ancestors = list(token.ancestors)
					if root is not '' and root in [str(ancestors[i]) for i in range(len(ancestors))]:
						motion_tokens.append(token.text)
					if root is not '' and token.tag_ in ["NN", "NNS", "RB", "JJ"]:
						root = ''
				if len(motion_tokens) < 2:
					continue
				motion = ' '.join(motion_tokens) + '.'
				train[id_]["motion"].append(motion)
				if train[id_]["motion"] == []:
					print(text)
					break


			train[id_]["ov_main_car"] = main_car
			train[id_]["main_car"].extend(main_car)
			train[id_]["ov_other_cars"] = other_cars
	return train

def extract_keywords(train):
	action_keyword_processor = load_keyword_processor("scripts/data/action_vocabulary.json")
	color_keyword_processor = load_keyword_processor("scripts/data/color_vocabulary.json")
	vehicle_keyword_processor = load_keyword_processor("scripts/data/vehicle_vocabulary.json")

	track_ids = list(train.keys())
	for id_ in tqdm(track_ids):
		new_text = ""
		for i, text in enumerate(train[id_]["nl"]):
			action = action_keyword_processor.extract_keywords(text)
			color = color_keyword_processor.extract_keywords(text)
			vehicle = vehicle_keyword_processor.extract_keywords(text)
			if action == [""] or color == [""] or vehicle == [""]:
				print(text)
			train[id_]["nl"][i] = ' '.join(color + vehicle + action)+'. '+train[id_]["nl"][i]
			new_text += ' '.join(color + vehicle + action)+'. '
			if i<2:
				new_text+=' '
		train[id_]["nl"].append(new_text)
	
	if USE_OTHER_VIREWS:
		for id_ in tqdm(track_ids):
			new_text = ""
			if train[id_]["nl_other_views"] == []:
				continue
			for i,text in enumerate(train[id_]["nl_other_views"]):
				action = action_keyword_processor.extract_keywords(text)
				color = color_keyword_processor.extract_keywords(text)
				vehicle = vehicle_keyword_processor.extract_keywords(text)
				if action == [""] or color == [""] or vehicle == [""]:
					print(text)
				train[id_]["nl_other_views"][i] = ' '.join(color + vehicle + action)+'. '+train[id_]["nl_other_views"][i]
				new_text += ' '.join(color + vehicle + action)+'. '


	return train

def extract_keywords_motion(train):
	action_keyword_processor = load_keyword_processor("scripts/data/action_vocabulary.json")
	color_keyword_processor = load_keyword_processor("scripts/data/color_vocabulary.json")
	vehicle_keyword_processor = load_keyword_processor("scripts/data/vehicle_vocabulary.json")

	track_ids = list(train.keys())
	for id_ in tqdm(track_ids):
		train[id_]["motion"] = []
		new_text = ""
		for i, text in enumerate(train[id_]["nl"]):
			action = action_keyword_processor.extract_keywords(text)
			color = color_keyword_processor.extract_keywords(text)
			vehicle = vehicle_keyword_processor.extract_keywords(text)
			if action == [""] or color == [""] or vehicle == [""]:
				print(text)
			train[id_]["nl"][i] = ' '.join(color + vehicle + action)+'. '+train[id_]["nl"][i]
			new_text += ' '.join(color + vehicle + action)+'. '
			if i<2:
				new_text+=' '
			
			# motion
			root = ''
			motion_tokens = []
			doc = nlp(text)
			for token in doc:
				if token.dep_ == "ROOT":
					root = token.text
					motion_tokens.append(token.text)
				ancestors = list(token.ancestors)
				if root is not '' and root in [str(ancestors[i]) for i in range(len(ancestors))]:
					motion_tokens.append(token.text)
				if root is not '' and token.tag_ in ["NN", "NNS", "RB", "JJ"]:
					
					break
			motion = ' '.join(motion_tokens) + '.'
			train[id_]["motion"].append(motion)

		train[id_]["nl"].append(new_text)
	
	if USE_OTHER_VIREWS:
		for id_ in tqdm(track_ids):
			new_text = ""
			if train[id_]["nl_other_views"] == []:
				continue
			for i,text in enumerate(train[id_]["nl_other_views"]):
				action = action_keyword_processor.extract_keywords(text)
				color = color_keyword_processor.extract_keywords(text)
				vehicle = vehicle_keyword_processor.extract_keywords(text)
				if action == [""] or color == [""] or vehicle == [""]:
					print(text)
				train[id_]["nl_other_views"][i] = ' '.join(color + vehicle + action)+'. '+train[id_]["nl_other_views"][i]
				new_text += ' '.join(color + vehicle + action)+'. '


	return train

def load_keyword_processor(data, is_path=True):
	if is_path:
		with open(data) as f:
			data = json.load(f)
	else:
		data = data
	keyword_processor = KeywordProcessor()
	for word in data:
		keyword_processor.add_keyword(word)
	return keyword_processor

def extract_noun(train):

	track_ids = list(train.keys())
	for id_ in tqdm(track_ids):
		new_text = ""
		for i,text in enumerate(train[id_]["nl"]):
			doc = nlp(text)

			for chunk in doc.noun_chunks:
				nb = chunk.text
				break
			train[id_]["nl"][i] = nb+'. '+train[id_]["nl"][i]
			new_text += nb+'.'
			if i<2:
				new_text+=' '
		train[id_]["nl"].append(new_text)
		
		if USE_OTHER_VIREWS:
			for i,text in enumerate(train[id_]["nl_other_views"]):
				doc = nlp(text)

				for chunk in doc.noun_chunks:
					nb = chunk.text
					break
				train[id_]["nl_other_views"][i] = nb+'. '+train[id_]["nl_other_views"][i]
				new_text += nb+'.'
				if i<2:
					new_text+=' '
			train[id_]["nl_other_views"].append(new_text)
	return train


def train_test_split(data_dict, test_prob=0.1, random_state=42):
	random.seed(random_state)
	train, test = {}, {}
	for k, v in data_dict.items():
		if random.random() > test_prob:
			train[k] = v
		else:
			test[k] = v
	return train, test

 
if __name__ == '__main__':

	with open("data/test_queries.json") as f:
		data = json.load(f)
	data = add_clean(data)		
	with open("data/test_queries.json".replace(".json", "_nlpaugv5.json"), "w") as f:
		json.dump(data, f,indent=4)
	
	with open("data/train_tracks.json") as f:
		data = json.load(f)
	data = add_clean(data)		
	with open("data/train_tracks.json".replace(".json", "_nlpaugv5.json"), "w") as f:
		json.dump(data, f,indent=4)
	

