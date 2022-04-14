<!--
 * @Date: 2022-04-14 15:34:16
 * @LastEditors: yhxiong
 * @LastEditTime: 2022-04-14 15:59:22
 * @Description: 
-->
# AI City 2021: Connecting Language and Vision for Natural Language-Based Vehicle Retrieval
🏆 The 1st Place Submission to AICity Challenge 2021 Natural Language-Based Vehicle Retrieval Track (Alibaba-UTS submission)

![framework](figs/framework.jpg)





## Prepare
-  Preprocess the dataset to prepare `frames, motion maps, NLP augmentation`

` scripts/extract_vdo_frms.py` is a Python script that is used to extract frames.

` scripts/get_motion_maps.py` is a Python script that is used to get motion maps.

- data augmentation
- 
` scripts/deal_nlpaug.py` is a Python script that is used for NLP augmentation.
` scripts/add_cnt.py` is a Python script that is used to count frequency.



- Pretrain videoclip or swin3D
TODO

- [TODO]Download the pretrained models. The checkpoints can be found [here](https://drive.google.com/drive/folders/1LAtP_CkNsM9ZDHlcr2PVmrR6f7YI-AQK?usp=sharing).

### Train
The configuration files are in `configs`.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --name your_experiment_name --config your_config_file 
```

### Test

Change the `RESTORE_FROM` in your configuration file.

```
python -u test.py --config your_config_file
```

[TODO]Extract the visual and text embeddings. The extracted embeddings can be found [here](https://drive.google.com/drive/folders/1DBVapSsw2glnJi_LxiRaIQXu3CWDfZbe?usp=sharing).



## Submission

During the inference, we average all the frame features of the target in each track as track features, the embeddings of text descriptions are also averaged as the query features. The cosine distance is used for ranking as the final result. 

- Reproduce the best submission. ALL extracted embeddings are in the folder `output`:

```
python scripts/get_submit.py
```


## visualize tool

Since we have no access to the test set, we use visualize tool to evaluate our predictions.

![visual](figs/visual.png)


## Friend Links：

- https://github.com/ShuaiBai623/AIC2021-T5-CLV
