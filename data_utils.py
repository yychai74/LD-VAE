from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import random
from transformers import AutoTokenizer, GPT2Tokenizer, AutoModel
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import os


def get_data(path, data_type):
    sents, labels, label_idx = [], [], []
    df = pd.read_csv(path, sep="	", header=None)

    sents_array = df.iloc[:, 1].values.tolist()
    label_array = df.iloc[:, 0].values.tolist()
    label_array = [eval(a) for a in label_array]

    for temp, sen in zip(label_array, sents_array):
        if "train" in data_type:
            if sum(temp) > 0:
                labels.append(temp)
            else:
                continue
        else:
            labels.append(temp)

        sen = sen.split()
        if sen != '':
            sents.append(" ".join(sen))
    # print(len(sents_array))
    # print(len(sents))
    return sents, labels


class EmotionDataset(Dataset):
    def __init__(self, tokenizer, data_type, data_dir, device, max_len, class_num, batch_size=16, first_cluster=False):
        self.data_path = f"data/{data_dir}/train.txt"
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.first_cluster = first_cluster
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.sentence_embedding_name = f"{data_dir}_content_embedding.pth"
        if not os.path.exists(self.sentence_embedding_name):
            self.bert = AutoModel.from_pretrained("bert-base-uncased")
            self.bert.to(device)
            self.device = device
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

        self.max_len = max_len
        self.data_dir = data_dir
        self.data_type = data_type
        self.target_length = max_len
        self.batch_size = batch_size
        self.content_embeddings = None
        # self.cluster_num = 1000 if data_dir == 'AAPD' else 500
        if data_dir == "AAPD":
            self.cluster_num = 1000
        elif data_dir == "ACSA":
            self.cluster_num = 1500
        else:
            self.cluster_num = 200
        self.class_num = class_num
        # self.prompt = "A sentence that expresses emotion " if data_dir == 'SemEvalEc' else "An abstract that relate to topic of "
        # self.prompt = "An abstract that relate to topic of " if data_dir == 'AAPD' else "A sentence that expresses emotion "
        if data_dir == 'SemEvalEc':
            self.prompt = "A sentence that expresses emotion "
        elif data_dir == "AAPD":
            self.prompt = "An abstract that relate to topic of "
        else:
            self.prompt = "A movie review that relate to "
        self.sep = " | " if data_dir == 'AAPD' else ", "

        self.inputs = []
        self.targets = []
        self.target_seq = []
        self.bert_inputs = []
        self.clusters = []
        # self.idx2label = {a: b for a, b in enumerate(
        #     ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise',
        #      'trust'])}
        if data_dir == "AAPD":
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
                             "cs.ir": 20, "quant-ph": 21, "cs.gt": 22, "cs.cy": 23, "cs.pl": 24, "cs.se": 25,
                             "math.pr": 26, "cs.db": 27, "cs.cg": 28, "cs.na": 29, "cs.hc": 30, "math.na": 31,
                             "cs.ce": 32, "cs.ma": 33, "cs.ro": 34, "cs.fl": 35, "math.st": 36, "stat.th": 37,
                             "cs.dl": 38, "cmp-lg": 39, "cs.mm": 40, "cond-mat.stat-mech": 41, "cs.pf": 42,
                             "math.lo": 43, "stat.ap": 44, "cs.ms": 45, "stat.me": 46, "cs.sc": 47,
                             "cond-mat.dis-nn": 48, "q-bio.nc": 49, "physics.data-an": 50, "nlin.ao": 51,
                             "q-bio.qm": 52, "math.nt": 53}
            self.idx2label = {b: topic_desc_map[a] for a, b in topic_num_map.items()}
        elif data_dir == "IMDB":
            labels = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                      'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
                      'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
                      'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
                      'Thriller', 'War', 'Western']
            self.idx2label = {a: b for a, b in enumerate(labels)}
        else:
            labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness',
                      'surprise', 'trust']
            self.idx2label = {a: b for a, b in enumerate(labels)}

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.target_seq[index]["input_ids"].squeeze()
        target = torch.tensor(self.targets[index])
        clusters = torch.tensor(self.clusters[index])
        bert_ids = self.bert_inputs[index]["input_ids"].squeeze()
        # target = self.targets[index]

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        bert_mask = self.bert_inputs[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "labels": target, "target_ids": target_ids,
                "bert_ids": bert_ids, "bert_mask": bert_mask, "cluster": clusters}

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _build_examples(self):
        # inputs, targets = get_transformed_data(self.data_path, self.data_dir, self.data_type)
        inputs, targets = get_data(self.data_path, self.data_type)

        print("Begin clustering ... ")
        if not os.path.exists(self.sentence_embedding_name):
            sentence_embeddings = []
            for example in tqdm(inputs):
                # Tokenize sentences
                encoded_input = self.bert_tokenizer([example], padding=True, truncation=True, max_length=self.max_len,
                                                    return_tensors='pt').to(self.device)

                # Compute token embeddings
                with torch.no_grad():
                    model_output = self.bert(**encoded_input)

                # Perform pooling. In this case, mean pooling
                sentence_embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
                # print(sentence_embedding.size())
                sentence_embeddings.append(sentence_embedding.cpu())
            sentence_embeddings = torch.cat(sentence_embeddings)
            print(sentence_embeddings.shape)
            torch.save(sentence_embeddings, self.sentence_embedding_name)
        else:
            sentence_embeddings = torch.load(self.sentence_embedding_name)

        km = KMeans(n_clusters=self.cluster_num, random_state=0, n_init="auto").fit(sentence_embeddings.numpy())
        sen2cluster = {sen: cluster for sen, cluster in zip(inputs, km.labels_)}
        cluster_centers = np.concatenate(km.cluster_centers_).reshape(self.cluster_num, -1)

        self.content_embeddings = torch.from_numpy(cluster_centers)
        # torch.save(label_num_embeddings, "clusters.pth")

        targets = [str(t) for t in targets]
        input2target = {b: a for a, b in zip(targets, inputs)}
        label_set = list(sorted(set(targets)))
        # random.shuffle(label_set)
        # print(len(label_set))
        temp_inputs, temp_labels = [], []

        label_same_dict = {l: [] for l in label_set}
        for sent in inputs:
            label_same_dict[input2target[sent]].append(sent)

        pad_seq = [self.tokenizer.pad_token] * 1
        pad_seq = " ".join(pad_seq)
        self.pad_seq = pad_seq
        pad_label = [0] * self.class_num

        # pad batch based on label composition
        for sent_label, sent_list in label_same_dict.items():
            if len(sent_list) % self.batch_size == 0:
                temp_inputs.append(sent_list)
                temp_labels.append([sent_label] * len(sent_list))
            else:
                left_num = len(sent_list) % self.batch_size
                origin_len = len(sent_list)
                sent_list += [pad_seq] * (self.batch_size - left_num)
                temp_inputs.append(sent_list)
                append_labels = [sent_label] * origin_len + [str(pad_label)] * (self.batch_size - left_num)
                temp_labels.append(append_labels)

        temp_inputs = sum(temp_inputs, [])
        temp_labels = sum(temp_labels, [])

        self.targets = [eval(a) for a in temp_labels]

        print("Tokenizing data...")
        for s, label in tqdm(zip(temp_inputs, self.targets)):
            label_words = self.sep.join([self.idx2label[idx] for idx, l in enumerate(label) if l == 1])
            input = self.prompt + label_words + " : " + s + " " + self.tokenizer.eos_token if label_words != "" else pad_seq
            target = self.prompt + label_words + " : " + s + " " + self.tokenizer.eos_token if label_words != "" else pad_seq

            bert_input = s if label_words != "" else self.bert_tokenizer.pad_token

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input], max_length=self.max_len, truncation=True, padding='max_length',
                return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len, truncation=True, padding='max_length',
                return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.target_seq.append(tokenized_target)
            self.bert_inputs.append(
                self.bert_tokenizer(bert_input, add_special_tokens=True, max_length=self.max_len, truncation=True,
                                    padding='max_length', return_tensors="pt"))
            if s in sen2cluster.keys():
                self.clusters.append(sen2cluster[s])
            else:
                self.clusters.append(-1)

