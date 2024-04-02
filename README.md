# Compositional Generalization for Multi-Label Text Classification: A Data-Augmentation Approach
This is the source code for AAAI 2024 paper: [Compositional Generalization for Multi-Label Text Classification: A Data-Augmentation Approach](https://ojs.aaai.org/index.php/AAAI/article/view/29725)
## 1. Environments

```
- python (3.9.5)
- cuda (12.0)
```

## 2. Dependencies
You need to install `torch (2.0.0)` firstly, and then install other dependencies with `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 3. Dataset
We provide our compositional data split and estimate label composition distribution by GPT2 in `\data` folder. You can train your own label generator on support with GPT2.

## 4. Training

We provide one-step training files, you can easily conduct experiments by running:

```bash
sh train_SemEval.sh
sh train_AAPD.sh
sh train_IMDB.sh
```

The experiments for the SemEval, AAPD, and IMDB datasets are estimated to respectively take approximately 40 minutes, 8 hours, and more than 12 hours, when using a single RTX4090 GPU for training.
Using 1 RTX 3090 may double the training time :).

## 5. Ciation

If you find our work useful for your application or reaserch, please kindly cite our paper:

```
@article{Chai_Li_Liu_Chen_Li_Ji_Teng_2024, 
author={Chai, Yuyang 
and Li, Zhuang 
and Liu, Jiahui 
and Chen, Lei 
and Li, Fei 
and Ji, Donghong 
and Teng, Chong}, 
title={Compositional Generalization for Multi-Label Text Classification: A Data-Augmentation Approach}, 
volume={38}, 
year={2024},
pages={17727-17735}}
```