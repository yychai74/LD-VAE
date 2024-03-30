import random
from abc import ABC
from transformers import GPT2LMHeadModel, GPT2PreTrainedModel, AutoTokenizer, BertModel
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn as nn
import numpy as np
import os
import math
from pprint import pprint
import copy

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BertClassificationModel(nn.Module):
    def __init__(self, class_num):
        super(BertClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.class_num = class_num
        self.ffn = nn.ModuleList(nn.Linear(768, 2) for _ in range(self.class_num))

    def forward(self, input_ids, attention_mask):
        hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        cls_vec = hidden_states[:, 0, :]
        pred_list = [torch.stack([ffn(cls_vec[j]) for ffn in self.ffn], dim=0) for j in
                     range(cls_vec.shape[0])]
        # print(pred_list.shape)
        pred_list = torch.stack(pred_list, dim=0)

        return pred_list


class LabelGRU(nn.Module):
    def __init__(self, label_num, input_dim):
        super().__init__()
        self.label_grus = nn.ModuleList(
            [nn.GRU(input_dim, input_dim // 2, batch_first=True, bidirectional=True) for _ in range(label_num)])

    def forward(self, inputs, attention_mask, label=None, train=True):
        length = attention_mask.sum(dim=1)
        packed_inputs = pack_padded_sequence(inputs, length.cpu(), batch_first=True, enforce_sorted=False)

        outs = []
        if train:
            for i in label:
                packed_outs2, h_n = self.label_grus[i](packed_inputs)
                out = torch.cat([h_n[-1], h_n[-2]], dim=-1)
                outs.append(out)
        else:
            for gru in self.label_grus:
                packed_outs2, h_n = gru(packed_inputs)
                out = torch.cat([h_n[-1], h_n[-2]], dim=-1)
                outs.append(out)

        return torch.stack(outs, dim=1)


class LD_VAE(GPT2PreTrainedModel):
    def __init__(self, config, mid_dim, tokenizer, dataset, preseqlen=20, use_adapter=False, content_embeddings=None):
        super().__init__(config)
        print('under the LD_VAE model')

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.preseqlen = preseqlen
        self.mid_dim = mid_dim
        self.dropout = nn.Dropout(0.0)
        self.use_adapter = use_adapter
        self.tokenizer = tokenizer
        self.base_config = config

        if dataset == 'SemEval':
            self.labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism',
                           'sadness', 'surprise', 'trust']
        elif dataset == 'AAPD':
            self.labels = list({
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
                               }.values())
        else:
            self.labels = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                           'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Game-Show',
                           'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
                           'Reality-TV', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show',
                           'Thriller', 'War', 'Western']
        self.idx2label = {a: b for a, b in enumerate(self.labels)}
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt", pad_token_id=tokenizer.pad_token_id)
        self.gpt.resize_token_embeddings(len(self.tokenizer))
        for param in self.gpt.base_model.parameters():
            param.requires_grad = False

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for n, param in self.bert.base_model.named_parameters():
            param.requires_grad = False

        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        low_data_init = 0
        print('[Full prefix-tuning Setting :) ]')
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, 768)
        self.content_wte = nn.Embedding.from_pretrained(content_embeddings, freeze=True)
        # print(self.content_wte.weight.shape)

        self.init_label_embedding()
        self.control_trans = nn.Sequential(
            nn.Linear(mid_dim * 2 + 768, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))

        self.mse_loss = nn.MSELoss()

        self.latent_size = mid_dim
        self.content_hidden2mean = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, mid_dim),
        )
        self.content_hidden2logv = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, mid_dim),
        )
        self.label_hidden2mean = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, mid_dim),
        )
        self.label_hidden2logv = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, mid_dim),
        )
        self.label_gaussian_hidden2mean = nn.Sequential(
            nn.Linear(config.n_embd, mid_dim)
        )
        self.content_gaussian_hidden2mean = nn.Sequential(
            nn.Linear(config.n_embd, mid_dim)
        )

        self.get_prompt = self.get_prompt_p5
        self.label_gru = LabelGRU(label_num=self.label_embedding.weight.shape[0],
                                  input_dim=self.bert.config.hidden_size)
        self.content_gru = nn.GRU(self.bert.config.hidden_size, self.bert.config.hidden_size // 2, bidirectional=True,
                                  batch_first=True)

    def init_label_embedding(self):
        label_embeddings = []
        for label in self.labels:
            emb = self.bert(**self.bert_tokenizer(label, return_tensors="pt", is_split_into_words=False,
                                                  add_special_tokens=False)).last_hidden_state
            # print(emb)
            label_embeddings.append(emb.data.mean(1, keepdim=True))
        wte = torch.cat(label_embeddings, dim=1).squeeze()
        # print(wte[0])
        self.label_embedding = nn.Embedding.from_pretrained(wte, freeze=True)
        print(self.label_embedding.weight.shape)

    def get_tokens(self, idx_tokens):
        start_idxes = np.multiply(np.asarray(idx_tokens), 10)
        end_idxes = np.multiply(np.asarray(idx_tokens) + 1, 10)
        input_tokens = []
        for s, e in zip(start_idxes, end_idxes):
            input_tokens.append(list(range(s, e)))
        input_tokens = sum(input_tokens, [])
        return input_tokens

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def orthogonal_penalty(self, label_mean, content_mean, l_n_norm=2):
        mm = torch.matmul(label_mean, content_mean.permute(1, 0))

        return torch.pow(mm, l_n_norm).mean()

    def get_prompt_p5(self, control_code=None, train=True, num_beams=None, bsz=None, labels=None, bert_ids=None,
                      bert_mask=None,
                      clusters=None):
        if labels is not None:
            labels = labels[0].data.cpu() if len(labels.shape) == 2 else labels
            # print(labels)
            idx_tokens = [idx for idx, l in enumerate(labels) if l == 1]
            # print(idx_tokens)
            label_num = len(idx_tokens)

            input_tokens = self.input_tokens.expand(bsz, -1).to(self.device)

            if train:
                hidden = self.bert(bert_ids, bert_mask).last_hidden_state
                label_v = self.label_gru(hidden, bert_mask, idx_tokens)

                length = bert_mask.sum(dim=1)
                packed_hidden = pack_padded_sequence(hidden, length.cpu(), batch_first=True, enforce_sorted=False)
                packed_hidden, _ = self.content_gru(packed_hidden)
                cls_v, _ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=bert_mask.shape[1])
                cls_v = self.mean_pooling(cls_v[:, :, :], bert_mask[:, :])

                # REPARAMETERIZATION
                cls_mean = self.content_hidden2mean(cls_v)
                cls_logv = self.content_hidden2logv(cls_v)
                label_mean = self.label_hidden2mean(label_v)
                label_logv = self.label_hidden2logv(label_v)

                # prior
                content_gauss = self.content_wte(clusters)
                content_mean = self.content_gaussian_hidden2mean(content_gauss)
                label_gauss = self.label_embedding(torch.tensor(idx_tokens, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1))
                label_gauss_mean = self.label_gaussian_hidden2mean(label_gauss)

                q_latent_loss1 = self.mse_loss(content_mean, cls_mean.detach())
                q_latent_loss2 = self.mse_loss(label_gauss_mean, label_mean.detach())
                e_latent_loss1 = self.mse_loss(content_mean.detach(), cls_mean)
                e_latent_loss2 = self.mse_loss(label_gauss_mean.detach(), label_mean)

                content_std = torch.exp(0.5 * cls_logv)
                label_std = torch.exp(0.5 * label_logv)

                z_content = torch.randn([bsz, self.latent_size], device=self.device)
                z_content = z_content * content_std + cls_mean
                z_content = z_content.unsqueeze(1)
                # print(z_content)

                z_label = torch.randn([bsz, len(idx_tokens) * 10, self.latent_size], device=self.device)
                z_label = (z_label * label_std.repeat(1, 1, 10).view(label_std.shape[0], 10 * len(idx_tokens), -1) +
                           label_mean.repeat(1, 1, 10).view(label_mean.shape[0], 10 * len(idx_tokens), -1))

                # a little different from the formula in our paper
                # this is for the stability of model training
                loss_content = 0.2 * q_latent_loss1 + 1 * e_latent_loss1 + torch.mean(content_std) - 1.0 - torch.mean(
                    cls_logv)
                loss_label = 0.2 * q_latent_loss2 + 1 * e_latent_loss2 + torch.mean(label_std) - 1.0 - torch.mean(
                    label_logv)

                loss2 = {"label": loss_label, "content": loss_content}
            else:
                content_gauss = self.content_wte(
                    torch.randint(self.content_wte.weight.shape[0], (bsz,), device=self.device))
                content_mean = self.content_gaussian_hidden2mean(content_gauss)

                label_gauss = self.label_embedding(
                    torch.tensor(idx_tokens, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1))
                label_gauss_mean = self.label_gaussian_hidden2mean(label_gauss)

                z_content = torch.randn([bsz, self.latent_size], device=self.device)
                z_content = z_content * 1 + content_mean
                z_content = z_content.unsqueeze(1)

                z_label = torch.randn([bsz, len(idx_tokens) * 10, self.latent_size],
                                      device=self.device)
                z_label = z_label * 1 + label_gauss_mean.repeat(1, 1, 10).view(label_gauss_mean.shape[0],
                                                                               10 * len(idx_tokens), -1)
                loss2 = None
        else:
            input_tokens = self.input_tokens.expand(bsz, -1).to(self.device)

        temp_control = self.wte(input_tokens)
        temp_control = temp_control.repeat(1, label_num, 1)

        temp_control = torch.cat([temp_control, torch.cat([z_label, z_content.repeat(1, z_label.shape[1], 1)], dim=-1)],
                                 dim=-1)

        past_key_values = self.control_trans(temp_control)
        past_key_values_mlp = past_key_values

        bsz, seqlen, _ = past_key_values.shape
        if num_beams is not None:
            past_key_values = past_key_values.repeat(1, num_beams, 1).view(bsz * num_beams, seqlen, -1)
            bsz = bsz * num_beams
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)

        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        if train:
            return past_key_values, loss2, cls_mean, label_mean
        else:
            return past_key_values, past_key_values_mlp.view(bsz, seqlen, self.match_n_layer * 2, -1)

    def forward(self,
                input_ids=None,
                weights=None,
                control_code=None,
                emb_match=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                src=None,
                tgt=None,
                src_attn=None,
                tgt_attn=None,
                label_list=None,
                bert_ids=None,
                bert_mask=None,
                clusters=None,
                **kwargs,
                ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]
        # self.active_adapter(label_list)
        past_key_values_prompt, loss2, _, _ = self.get_prompt(control_code, bsz=bsz, labels=label_list,
                                                              clusters=clusters,
                                                              bert_ids=bert_ids, bert_mask=bert_mask, train=True)
        # past_key_values_prompt = self.get_prompt(control_code, gpt2=self.gpt, bsz=bsz)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        output = self.gpt(input_ids=input_ids,
                          past_key_values=past_key_values,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids, position_ids=position_ids,
                          head_mask=head_mask, inputs_embeds=inputs_embeds,
                          encoder_hidden_states=encoder_hidden_states,
                          encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                          output_attentions=output_attentions, output_hidden_states=True,
                          return_dict=return_dict, **kwargs)

        return output, loss2


class LS_PT(GPT2PreTrainedModel):
    def __init__(self, config, tokenizer, preseqlen=20):
        super().__init__(config)
        print('under the LS_PT model')

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.preseqlen = preseqlen
        self.dropout = nn.Dropout(0.0)
        self.tokenizer = tokenizer
        self.base_config = config

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt", pad_token_id=tokenizer.pad_token_id,
                                                   output_hidden_states=True)
        self.gpt.resize_token_embeddings(len(self.tokenizer))
        for param in self.gpt.base_model.parameters():
            param.requires_grad = False

        low_data_init = 0
        print('[Full prefix-tuning Setting :) ]')
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, config.n_embd)

        self.control_trans = nn.Sequential(
            nn.Linear(config.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))
        self.get_prompt = self.get_prompt_p5

    def get_tokens(self, idx_tokens):
        start_idxes = np.multiply(np.asarray(idx_tokens), 10)
        end_idxes = np.multiply(np.asarray(idx_tokens) + 1, 10)
        input_tokens = []
        for s, e in zip(start_idxes, end_idxes):
            input_tokens.append(list(range(s, e)))
        input_tokens = sum(input_tokens, [])
        return input_tokens

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_prompt_p5(self, control_code=None, train=True, bsz=None, labels=None):
        if labels is not None:
            labels = labels[0].data.cpu() if len(labels.shape) == 2 else labels
            idx_tokens = [idx for idx, l in enumerate(labels) if l == 1]

            input_tokens = self.get_tokens(idx_tokens)
            input_tokens = torch.from_numpy(np.asarray(input_tokens)).unsqueeze(0).expand(bsz, -1).to(self.device)
            # print(input_tokens)
        else:
            input_tokens = self.input_tokens.expand(bsz, -1).to(self.device)

        temp_control = self.wte(input_tokens)
        past_key_values_mlp = self.control_trans(temp_control)

        bsz, seqlen, _ = past_key_values_mlp.shape
        past_key_values = past_key_values_mlp.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                   self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values

    def forward(self,
                input_ids=None,
                weights=None,
                control_code=None,
                emb_match=None,
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                src=None,
                tgt=None,
                src_attn=None,
                tgt_attn=None,
                label_list=None,
                bert_ids=None,
                bert_mask=None,
                clusters=None,
                **kwargs,
                ):

        # {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]
        past_key_values_prompt = self.get_prompt(control_code, bsz=bsz, labels=label_list)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt
        output = self.gpt(input_ids=input_ids,
                          past_key_values=past_key_values,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids, position_ids=position_ids,
                          head_mask=head_mask, inputs_embeds=inputs_embeds,
                          encoder_hidden_states=encoder_hidden_states,
                          encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                          output_attentions=True, output_hidden_states=output_hidden_states,
                          return_dict=return_dict, **kwargs)

        return output
