from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import random
from transformers import BertTokenizer
from itertools import combinations
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from tqdm import tqdm
import os
from collections import Counter
from pprint import pprint


def get_data_and_label(data_dir, data_type):
    df = pd.read_csv(f'data/{data_dir}/{data_type}.txt', sep="\t", header=None)
    data = df.iloc[:, 1].values.tolist()
    labels = df.iloc[:, 0].values.tolist()

    return data, labels


def get_data(data_dir, data_type, train_filter=False):
    sep = " | " if data_dir == "AAPD" else ", "
    # print(data_dir)

    train_data, train_label = get_data_and_label(data_dir, "train")
    test_data, test_label = get_data_and_label(data_dir, "test")
    support_data, support_label = get_data_and_label(data_dir, "support")

    train_label = [eval(label) for label in train_label]
    test_label = [eval(label) for label in test_label]
    support_label = [eval(label) for label in support_label]
    # print(train_data[0])
    # print(train_label[0])

    if not train_filter:
        aug_df = pd.read_csv(f"augmentation_data/aug_{data_dir}.txt", sep="\t", header=None)
        aug_data = aug_df.iloc[:, 1].values.tolist()
        aug_data = [" ".join(data.split(":")[1:]).strip() for data in aug_data]
        # print(aug_data[0])
        aug_label = aug_df.iloc[:, 0].values.tolist()
    else:
        aug_data = None

    if data_dir == "SemEval":
        label2idx = {label: idx for idx, label in enumerate(
            ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness',
             'surprise', 'trust'])}
    elif data_dir == "AAPD":
        topic_desc_map = {
            "cs.it": "Information Theory and Coding Theory",
            "math.it": "Information Theory",
            "cs.lg": "Machine Learning",
            "cs.ai": "Artificial Intelligence",
            "stat.ml": "Machine Learning and Statistics",
            "cs.ds": "Data Structures and Algorithms",
            "cs.si": "Social and Information Networks",
            "cs.dm": "Discrete Mathematics",
            "physics.soc-ph": "Physics and Society",
            "cs.lo": "Logic in Computer Science",
            "math.co": "Combinatorics",
            "cs.cc": "Computational Complexity",
            "math.oc": "Optimization and Control",
            "cs.ni": "Networking and Internet Architecture",
            "cs.cv": "Computer Vision and Pattern Recognition",
            "cs.cl": "Computation and Language (Natural Language Processing)",
            "cs.cr": "Cryptography and Security",
            "cs.sy": "Systems and Control",
            "cs.dc": "Distributed, Parallel, and Cluster Computing",
            "cs.ne": "Neural and Evolutionary Computing",
            "cs.ir": "Information Retrieval",
            "quant-ph": "Quantum Physics",
            "cs.gt": "Computer Science and Game Theory",
            "cs.cy": "Computational Geometry",
            "cs.pl": "Programming Languages",
            "cs.se": "Software Engineering",
            "math.pr": "Probability",
            "cs.db": "Databases",
            "cs.cg": "Computer Graphics",
            "cs.na": "Numerical Analysis",
            "cs.hc": "Human-Computer Interaction",
            "math.na": "Math Numerical Analysis",
            "cs.ce": "Computational Engineering, Finance, and Science",
            "cs.ma": "Multiagent Systems",
            "cs.ro": "Robotics",
            "cs.fl": "Formal Languages and Automata Theory",
            "math.st": "Statistics Theory",
            "stat.th": "Statistics Theory and Methods",
            "cs.dl": "Deep Learning",
            "cmp-lg": "Computational Linguistics",
            "cs.mm": "Multimedia",
            "cond-mat.stat-mech": "Statistical Mechanics and Condensed Matter",
            "cs.pf": "Performance",
            "math.lo": "Logic",
            "stat.ap": "Applied Statistics",
            "cs.ms": "Mathematical Software",
            "stat.me": "Methodology and Experimentation",
            "cs.sc": "Symbolic Computation",
            "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
            "q-bio.nc": "Neurons and Cognition",
            "physics.data-an": "Data Analysis, Statistics and Probability in Physics",
            "nlin.ao": "Adaptation and Self-Organizing Systems",
            "q-bio.qm": "Quantitative Methods",
            "math.nt": "Number Theory"
        }
        topic_num_map = {"cs.it": 0, "math.it": 1, "cs.lg": 2, "cs.ai": 3, "stat.ml": 4, "cs.ds": 5, "cs.si": 6,
                         "cs.dm": 7, "physics.soc-ph": 8, "cs.lo": 9, "math.co": 10, "cs.cc": 11, "math.oc": 12,
                         "cs.ni": 13, "cs.cv": 14, "cs.cl": 15, "cs.cr": 16, "cs.sy": 17, "cs.dc": 18, "cs.ne": 19,
                         "cs.ir": 20, "quant-ph": 21, "cs.gt": 22, "cs.cy": 23, "cs.pl": 24, "cs.se": 25, "math.pr": 26,
                         "cs.db": 27, "cs.cg": 28, "cs.na": 29, "cs.hc": 30, "math.na": 31, "cs.ce": 32, "cs.ma": 33,
                         "cs.ro": 34, "cs.fl": 35, "math.st": 36, "stat.th": 37, "cs.dl": 38, "cmp-lg": 39, "cs.mm": 40,
                         "cond-mat.stat-mech": 41, "cs.pf": 42, "math.lo": 43, "stat.ap": 44, "cs.ms": 45,
                         "stat.me": 46, "cs.sc": 47, "cond-mat.dis-nn": 48, "q-bio.nc": 49, "physics.data-an": 50,
                         "nlin.ao": 51, "q-bio.qm": 52, "math.nt": 53}
        label2idx = {topic_desc_map[label]: idx for label, idx in topic_num_map.items()}
    else:
        labels = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
                  'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
                  'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
                  'Thriller', 'War', 'Western']
        label2idx = {b: a for a, b in enumerate(labels)}

    aug_sents, aug_labels = [], []

    # simple uniform oversampling
    for sen, l in zip(support_data, support_label):
        for _ in range(2):
            train_data.append(sen)
            train_label.append(l)

    if aug_data:
        for aug_sent, aug_l in zip(aug_data, aug_label):
            temp_label = [0] * len(label2idx)
            for label in aug_l.split(sep):
                temp_label[label2idx[label]] = 1
            aug_labels.append(temp_label)

            # print(aug_sent)
            sent = aug_sent.split()
            if sent != '':
                aug_sents.append(" ".join(sent))

    print(len(test_data))
    # data split trick for SemEval
    if data_dir == "SemEval":
        if data_type == "train":
            return train_data + aug_sents, train_label + aug_labels
        elif data_type == "dev":
            return [], []
        else:
            return test_data, test_label
    else:
        if data_type == "train":
            return train_data, train_label
        elif data_type == "dev":
            return aug_sents, aug_labels
        else:
            return test_data, test_label


class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, data_type, data_dir, max_len=128, train_filter=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_dir = data_dir
        self.data_type = data_type
        self.target_length = max_len
        self.train_filter = train_filter

        self.inputs = []
        self.targets = None

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target = torch.tensor(self.targets[index])

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "labels": target}

    def _build_examples(self):
        inputs, self.targets = get_data(self.data_dir, self.data_type, self.train_filter)

        print("tokenizing data...")
        for i in tqdm(range(len(inputs))):
            input = inputs[i].strip()

            tokenized_input = self.tokenizer(
                input, max_length=self.max_len, padding='max_length', truncation=True,
                return_tensors="pt",
            )

            self.inputs.append(tokenized_input)
