import os
import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import get_linear_schedule_with_warmup, BertTokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import argparse
from model import LD_VAE, BertClassificationModel, LS_PT
from data_utils import EmotionDataset
import logging
import json
from tqdm import tqdm
from sklearn.metrics import jaccard_score
import random
import numpy as np
import pandas as pd
import itertools
from collections import Counter
from new_generate_utils import GenerationMixin
import types
from pprint import pprint


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='AAPD', type=str)
    parser.add_argument("--model", default='LS_PT', type=str)  # LS_PT

    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--eval", action='store_true', default=False)
    # parser.add_argument("--num_train_epochs", default=50, type=int)

    parser.add_argument("--n_gpu", default=[0])
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    if not os.path.exists('./augmentation_data'):
        os.mkdir('./augmentation_data')

    if not os.path.exists('./output'):
        os.mkdir('./output')

    # output_dir = f"./output/{args.dataset}/own"
    output_dir = f"./output/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    args.output_dir = output_dir
    args.device = args.n_gpu[0]
    args.num_train_epochs = 50 if args.model == "LD_VAE" else 20
    args.sep = " | " if args.dataset == "AAPD" else ", "

    return args


class Model(pl.LightningModule):
    def __init__(self, bert_learning_rate, other_learning_rate, adam_epsilon, warmup_steps, weight_decay,
                 train_batch_size, val_batch_size, class_num, max_length, gradient_accumulation_steps, alpha, beta):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.tokenizer = gpt_tokenizer
        if args.model == "LD_VAE":
            self.model = LD_VAE(config=GPT2Config.from_pretrained("gpt2"), preseqlen=10, mid_dim=512,
                                dataset=args.dataset,
                                tokenizer=self.tokenizer, use_adapter=False,
                                content_embeddings=train_dataset.content_embeddings)
        elif args.model == "LS_PT":
            self.model = LS_PT(config=GPT2Config.from_pretrained("gpt2"), preseqlen=self.hparams.class_num * 10,
                               tokenizer=self.tokenizer)
        else:
            raise ValueError("No such model")

        print(sum(x.numel() for x in self.model.parameters() if x.requires_grad))

    def forward(self, batch):
        labels = batch['target_ids']
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        attention_mask = batch['source_mask']
        pad_seq_mask = torch.sum(attention_mask[:, 1:], dim=1).gt(0)
        input_ids = batch['source_ids'][pad_seq_mask]
        labels = labels[pad_seq_mask]
        label_list = batch["labels"][pad_seq_mask]

        if args.model == "LD_VAE":
            bert_ids = batch["bert_ids"][pad_seq_mask]
            bert_mask = batch["bert_mask"][pad_seq_mask]
            clusters = batch["cluster"][pad_seq_mask]

            outputs, loss2 = self.model(input_ids, labels=labels, label_list=label_list,
                                        bert_ids=bert_ids, bert_mask=bert_mask, clusters=clusters)

            return outputs, loss2
        else:
            outputs = self.model(input_ids, labels=labels, label_list=label_list)

            return outputs

    def LD_VAE_step(self, batch):
        outputs, loss2 = self(batch)
        loss = outputs.loss
        label = loss2["label"]
        content = loss2["content"]

        return loss + self.hparams.alpha * label + self.hparams.beta * content, label, content

    def LS_PT_step(self, batch):
        outputs = self(batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        if args.model == "LD_VAE":
            loss, _, _ = self.LD_VAE_step(batch)
            return loss
        else:
            loss = self.LS_PT_step(batch)
            return loss

    def validation_step(self, batch, batch_idx):
        if args.model == "LD_VAE":
            loss, loss2, loss3 = self.LD_VAE_step(batch)
            return {'loss': loss, 'loss2': loss2, 'loss3': loss3}
        else:
            loss = self.LS_PT_step(batch)
            return {'loss': loss}

    def validation_epoch_end(self, outputs):
        if args.model == "LD_VAE":
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            avg_loss2 = torch.stack([x['loss2'] for x in outputs]).mean()
            avg_loss3 = torch.stack([x['loss3'] for x in outputs]).mean()

            self.log('val_loss', avg_loss)
            print(f'\nValidation loss: {avg_loss}')
            print(f'label loss: {avg_loss2}')
            print(f'content loss: {avg_loss3}')
        else:
            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

            self.log('val_loss', avg_loss)
            print(f'\nValidation loss: {avg_loss}')

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                'lr': self.hparams.bert_learning_rate
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                'lr': self.hparams.bert_learning_rate
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon,
                          lr=self.hparams.other_learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0.03 * self.trainer.estimated_stepping_batches,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def train_dataloader(self):
        print("train set size:", len(train_dataset))
        return DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, num_workers=8, shuffle=False)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=self.hparams.train_batch_size, shuffle=False, num_workers=8)


if __name__ == '__main__':
    args = init_args()
    torch.set_float32_matmul_precision('high')
    with open(f"configs/{args.dataset}/config.json", encoding='utf-8') as f:
        config = json.load(f)
    print("Model configs:")
    pprint(config)
    seed_everything(args.seed)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    special_tokens_dict = {"pad_token": "[PAD]"}
    gpt_tokenizer.add_special_tokens(special_tokens_dict)
    gpt_tokenizer.padding_side = 'right'

    train_dataset = EmotionDataset(tokenizer=gpt_tokenizer, data_type="train", data_dir=args.dataset,
                                   max_len=config["max_length"], batch_size=config["train_batch_size"],
                                   device=torch.device(f'cuda:{args.device}'),
                                   class_num=config["class_num"])
    val_dataset = train_dataset

    if args.train:
        print("*" * 20, f"Start Training on {args.dataset}", "*" * 20)

        model = Model(**config)
        callback = pl.callbacks.ModelCheckpoint(
            dirpath=args.output_dir, monitor='val_loss', mode='min', save_top_k=1, filename='{epoch}-' + f'{args.model}',
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
            callbacks=[callback, EarlyStopping(monitor="val_loss", patience=3, mode="min")],
            logger=log,
            check_val_every_n_epoch=10
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

    if args.eval:
        val_dataset = train_dataset
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
            label2index = {topic_desc_map[a]: b for a, b in topic_num_map.items()}
        elif args.dataset == "IMDB":
            labels = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                      'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
                      'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
                      'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
                      'Thriller', 'War', 'Western']
            label2index = {b: a for a, b in enumerate(labels)}
        else:
            emotion_labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism',
                              'sadness', 'surprise', 'trust']
            label2index = {l: i for i, l in enumerate(emotion_labels)}

        all_checkpoints = []

        df = pd.read_csv(f"data/{args.dataset}/labels.txt", sep="\t", header=None)
        labels = df.iloc[:, 0].values.tolist()
        emotion_prompts = [" ".join(lab.split()[9:]) for lab in labels] if args.dataset == "AAPD" else [
            " ".join(lab.split()[7:]) for lab in labels]

        prompts_counter = dict(Counter(emotion_prompts))
        pprint(prompts_counter)
        emotion_prompts_set = sorted(list(set(emotion_prompts)))
        label_lists = []

        for prompt in emotion_prompts:
            ori_list = [0] * config["class_num"]
            for label in prompt.split(args.sep):
                ori_list[label2index[label]] = 1
            label_lists.append(ori_list)

        for f in os.listdir(args.output_dir):
            file_name = os.path.join(args.output_dir, f)
            if f'{args.model}' in file_name:
                all_checkpoints.append(file_name)
        print(f'\nTest model on following checkpoints: {all_checkpoints}')

        device = torch.device(f'cuda:{args.device}')
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        data_filter = BertClassificationModel(config["class_num"])

        data_filter.load_state_dict(torch.load(f"{args.dataset}_filter.pth"))
        data_filter = data_filter.to(device)

        for ckpt in all_checkpoints:
            model = Model.load_from_checkpoint(f'{ckpt}', **config)
            model.to(device)
            model.eval()
            model.model.gpt.generate = types.MethodType(GenerationMixin.generate, model.model.gpt)
            model.model.gpt.sample = types.MethodType(GenerationMixin.sample, model.model.gpt)
            model.model.gpt.beam_search = types.MethodType(GenerationMixin.beam_search, model.model.gpt)

            print(f'results on {ckpt}')
            final = []
            final_dict = {}

            with torch.no_grad():
                for p, total_num in prompts_counter.items():
                    print("Generating sentence with label:", p)
                    prompt_list = []
                    label_list = [0] * config["class_num"]
                    for label in p.split(args.sep):
                        label_list[label2index[label]] = 1
                    for idx in tqdm(range(3 * total_num // 10 + 1)):
                        if args.dataset == 'SemEval':
                            input_seq = "A sentence that expresses emotion " + p + " :"
                        elif args.dataset == "AAPD":
                            input_seq = "An abstract that relate to topic of " + p + " :"
                        else:
                            input_seq = "A movie review that relate to " + p + " :"
                        # print(input_seq)
                        tokenized = gpt_tokenizer(input_seq, return_tensors='pt').to(device)
                        input_ids = tokenized["input_ids"].repeat(10, 1).to(device)
                        if args.model == "LD_VAE":
                            # we do not use beam search during generation, but the function is provided
                            prefix, _ = model.model.get_prompt(bsz=10, labels=torch.tensor(label_list), train=False,
                                                               num_beams=None)
                        else:
                            # beam search is not applicable for LS-PT
                            prefix = model.model.get_prompt(bsz=10, labels=torch.tensor(label_list))
                        beam_outputs = model.model.gpt.generate(
                            input_ids,
                            max_length=config["max_length"],
                            do_sample=True,
                            top_p=0.95,
                            top_k=50,
                            no_repeat_ngram_size=2,
                            num_return_sequences=1,
                            prefix=prefix,
                        )
                        prompt_list.append(
                            [p + "\t" + model.tokenizer.decode(beam_output, skip_special_tokens=True).replace("\n", "").strip()
                             for beam_output in beam_outputs])

                    final_dict[p] = sum(prompt_list, [])

                sorted_dicts = {}

                # filter
                for p, sen_list in final_dict.items():
                    temp = {}
                    label_list = [0] * config["class_num"]
                    for label in p.split(args.sep):
                        label_list[label2index[label]] = 1
                    for sen in sen_list:
                        tokenized = bert_tokenizer(" ".join(sen.split(" :")[1:]).strip(),
                                                   return_tensors="pt", max_length=config["max_length"],
                                                   padding='max_length')
                        out = data_filter(tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device))
                        pred = np.argmax(out.squeeze().data.cpu().numpy(), axis=-1).flatten().tolist()
                        temp[sen] = jaccard_score(label_list, pred)

                    temp = sorted(temp.items(), key=lambda x: x[1], reverse=True)
                    p_list = [(p, x[0]) for x in temp]
                    sorted_dicts[p] = p_list

                for l, num in prompts_counter.items():
                    final.append(sorted_dicts[l][: num])

                final = sum(final, [])
                print(len(final))
                random.shuffle(final)
                with open(f"augmentation_data/aug_{args.dataset}.txt", 'w') as w:
                    for p, generated in final:
                        w.write(generated + "\n")

