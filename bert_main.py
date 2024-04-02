import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import get_linear_schedule_with_warmup, BertTokenizer
import argparse
from model import BertClassificationModel
from bert_data_utils import ClassificationDataset
import logging
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, jaccard_score
import random
import numpy as np
import collections
from pprint import pprint
from torch import multiprocessing

multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='AAPD', type=str)

    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--train_filter", action='store_true', default=False)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    # parser.add_argument("--max_len", default=128, type=int)

    parser.add_argument("--n_gpu", default=[0])
    parser.add_argument("--seed", default=43, type=int)

    args = parser.parse_args()
    if not os.path.exists('./output'):
        os.mkdir('./output')

    if not os.path.exists(f'./output/{args.dataset}'):
        os.mkdir(f'./output/{args.dataset}')

    output_dir = f"./output/{args.dataset}/{args.seed}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    args.output_dir = output_dir
    args.device = args.n_gpu[0]

    return args


class Model(pl.LightningModule):
    def __init__(self, bert_learning_rate, other_learning_rate, adam_epsilon, warmup_steps, weight_decay,
                 train_batch_size, val_batch_size, class_num, max_length, gradient_accumulation_steps):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertClassificationModel(self.hparams.class_num)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        outputs = self.model(batch['source_ids'], batch['source_mask'])

        return outputs

    def _step(self, batch):
        outputs = self(batch)
        loss = self.loss_fn(outputs.view(-1, outputs.shape[-1]), batch["labels"].view(-1))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('val_loss', avg_loss)
        print(f'\nValidation loss: {avg_loss}')

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                'lr': self.hparams.bert_learning_rate
            },
            {
                "params": [p for n, p in self.model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                'lr': self.hparams.bert_learning_rate
            },
            {
                "params": self.model.ffn.parameters(),
                "weight_decay": 0.0,
                'lr': self.hparams.other_learning_rate
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon,
                          lr=self.hparams.other_learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def train_dataloader(self):
        print(len(train_dataset))
        return DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        print(len(val_dataset))
        return DataLoader(val_dataset, batch_size=self.hparams.val_batch_size, shuffle=False, num_workers=8)


if __name__ == '__main__':
    args = init_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision('high')
    with open(f"configs/{args.dataset}/bert_config.json", encoding='utf-8') as f:
        config = json.load(f)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.dataset == "SemEval":
        train_dataset = ClassificationDataset(tokenizer=tokenizer, data_type="train", data_dir=args.dataset, max_len=config["max_length"], train_filter=args.train_filter)
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1], generator=generator)
    else:
        train_dataset = ClassificationDataset(tokenizer=tokenizer, data_type="train", data_dir=args.dataset,
                                              max_len=config["max_length"], train_filter=args.train_filter)
        val_dataset = ClassificationDataset(tokenizer=tokenizer, data_type="dev", data_dir=args.dataset, max_len=config["max_length"], train_filter=args.train_filter)
        generator = torch.Generator().manual_seed(args.seed)
        aug_train, val_dataset1 = random_split(val_dataset, [0.95, 0.05], generator=generator)
        train_dataset, val_dataset2 = random_split(train_dataset, [0.95, 0.05], generator=generator)
        train_dataset = ConcatDataset([train_dataset, aug_train])
        val_dataset = ConcatDataset([val_dataset1, val_dataset2])

    if args.train:
        print("*" * 20, f"Start Training on {args.dataset}", "*" * 20)
        model = Model(**config)
        filename = "classifier" if not args.train_filter else "filter"
        callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir, monitor='val_loss', mode='min', save_top_k=2, filename="{epoch}-" + f"{filename}_{args.dataset}"
        )
        log = TensorBoardLogger('logs', name=args.dataset)

        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=config['gradient_accumulation_steps'],
            devices=args.n_gpu,
            accelerator="gpu",
            # strategy="ddp",
            gradient_clip_val=1.0,
            # amp_level='O1',
            max_epochs=args.num_train_epochs,
            callbacks=[callback, EarlyStopping(monitor="val_loss", patience=1, mode="min")],
            logger=log
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

    if args.eval:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        test_dataset = ClassificationDataset(tokenizer=tokenizer, data_type="test", data_dir=args.dataset,
                                             max_len=config["max_length"], train_filter=args.train_filter)
        print(len(test_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        all_checkpoints = []
        filename = "classifier" if not args.train_filter else "filter"

        if args.dataset == "AAPD":
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
                "math.na": "Numerical Analysis",
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
            idx2label = {b: topic_desc_map[a] for a, b in topic_num_map.items()}
        elif args.dataset == "IMDB":
            labels = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                      'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
                      'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
                      'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
                      'Thriller', 'War', 'Western']
            idx2label = {a: b for a, b in enumerate(labels)}

        else:
            labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness',
                      'surprise', 'trust']
            idx2label = {a: b for a, b in enumerate(labels)}


        def get_str_label(label_list):
            strs = [idx2label[idx] for idx, l in enumerate(label_list) if l == 1]
            return ", ".join(strs)


        for f in os.listdir(args.output_dir):
            file_name = os.path.join(args.output_dir, f)
            if f"{filename}_{args.dataset}" in file_name:
                all_checkpoints.append(file_name)
        print(f'\nTest model on following checkpoints: {all_checkpoints}')
        zz = 0.0
        for ckpt in all_checkpoints:
            model = Model.load_from_checkpoint(f'{ckpt}', **config)
            if args.train_filter:
                torch.save(model.model.state_dict(), f"{args.dataset}_filter.pth")
            device = torch.device(f'cuda:{args.device}')
            model.model.to(device)
            model.model.eval()

            preds, targets = [], []
            t = 0
            correctness, completeness = 0, 0
            sample_p, sample_r = [], []
            print(f'results on {ckpt}')

            for batch in tqdm(test_dataloader):
                with torch.no_grad():
                    out = model.model(batch['source_ids'].to(device), batch['source_mask'].to(device))
                    preds.append(np.argmax(out.data.cpu().numpy(), axis=-1))
                    targets.append(batch["labels"].cpu().numpy())

            outs = np.concatenate(preds, axis=0)
            gold = np.concatenate(targets, axis=0)

            gold_list = [get_str_label(g) for g in gold.tolist()]
            # print(gold_list)
            pprint(collections.Counter(gold_list))
            counter_gold = dict(collections.Counter(gold_list))
            counter_dict = {label: 0 for label in counter_gold.keys()}

            for a, b in zip(outs, gold):
                if np.all(a == b):
                    # print(c)
                    t += 1
                    counter_dict[get_str_label(b)] += 1
                if np.sum(a) != 0:
                    if np.all(np.logical_and(a, b) == b):
                        completeness += 1
                    if np.all(np.logical_and(a, b) == a):
                        correctness += 1
            acc = t / len(test_dataset)
            pprint(counter_dict)

            print(classification_report(gold, outs, digits=4))
            print(f"jaccard score: {jaccard_score(gold, outs, average='samples')}")
            print("accuracy:", acc)
            print("correctness:", correctness / len(test_dataset))
            print("completeness:", completeness / len(test_dataset))

