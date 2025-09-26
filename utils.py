# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Dandan Zheng, 2024/12/09
"""
import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data
import re
from Bio.PDB import PDBParser
from torch_geometric.data import Data
import scipy.sparse as sp
import random
MAX_INT = np.iinfo(np.int32).max


def read_preprocess_4pretrained(data_dir, filename, model_type="Bert"):
    f_fasta = open(data_dir + filename, "r")
    sequences_list = []
    for eachline in f_fasta.readlines():
        eachline = eachline.strip()
        if eachline.find(">") < 0:
            if model_type == "Bert" or model_type == "BioBERT" or model_type == "ProtBert" or model_type == "ProtT5":
                sequence = " ".join(eachline)
            else:
                sequence = eachline
            sequence = re.sub(r"[UZOB]", "X", sequence)
            sequences_list.append(sequence)
    return sequences_list


def read_preprocess_4pretrained_4predict(data_dir, filename, model_type="Bert"):
    f_fasta = open(data_dir + filename, "r")
    sequences_list = []
    ids_list = []
    for eachline in f_fasta.readlines():
        eachline = eachline.strip()
        if eachline.find(">") < 0:
            if model_type == "Bert" or model_type == "BioBERT" or model_type == "ProtBert" or model_type == "ProtT5":
                sequence = " ".join(eachline)
            else:
                sequence = eachline
            sequence = re.sub(r"[UZOB]", "X", sequence)
            sequences_list.append(sequence)
        else:
            ids_list.append(eachline)
    return ids_list, sequences_list


def read_preprocess_4pretrained_4terminal(data_dir, filename, model_type="Bert", terminal="Cterminal", maxlen=1022):
    f_fasta = open(data_dir + filename, "r")
    sequences_list = []
    for eachline in f_fasta.readlines():
        eachline = eachline.strip()
        if eachline.find(">") < 0:
        # 当terminal 为Cterminal 时，截断数据：
            if terminal == "Cterminal": 
                valid_each = eachline[-maxlen:]
            else:
                valid_each = eachline[:maxlen]
            
            if model_type == "Bert" or model_type == "BioBERT" or model_type == "ProtBert" or model_type == "ProtT5":
                sequence = " ".join(valid_each)
            else:
                sequence = valid_each
            sequence = re.sub(r"[UZOB]", "X", sequence)
            sequences_list.append(sequence)
    return sequences_list


def read_preprocess_4pretrained_4terminal_4predict(data_dir, filename, model_type="Bert", terminal="Cterminal", maxlen=1022):
    f_fasta = open(data_dir + filename, "r")
    sequences_list = []
    ids_list = []
    for eachline in f_fasta.readlines():
        eachline = eachline.strip()
        if eachline.find(">") < 0:
            # 当terminal 为Cterminal 时，截断数据：
            if terminal == "Cterminal":
                valid_each = eachline[-maxlen:]

            if model_type == "Bert" or model_type == "BioBERT" or model_type == "ProtBert" or model_type == "ProtT5":
                sequence = " ".join(valid_each)
            else:
                sequence = valid_each
            sequence = re.sub(r"[UZOB]", "X", sequence)
            sequences_list.append(sequence)
        else:
            ids_list.append(eachline)
    return ids_list, sequences_list


def read_preprocess_4pretrained_4numpy(sequences, model_type="Bert"):
    final_seq = []
    for eachseq in sequences:
        if model_type == "Bert" or model_type == "BioBERT" or model_type == "ProtBert" or model_type == "ProtT5":
            sequence = " ".join(eachseq)
        else:
            sequence = eachseq
        sequence = re.sub(r"[UZOB]", "X", sequence)
        final_seq.append(sequence)
    return np.array(final_seq)


def extract_features(sequence, pretrained_type, model, tokenizer, device, max_length=512):
    inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        if pretrained_type == "ProtT5":
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        else:
            outputs = model(**inputs)

    features = outputs.last_hidden_state
    features = features.cpu()
    return features


def batch_extract_features(sequences, pretrained_type, model, tokenizer, device, max_length=512, batch_size=10):
    """
    批量提取序列特征，并返回所有序列的特征张量和 attention_mask。
    Args:
        sequences (list): 序列列表。
        pretrained_type: pretrained model type, model's name
        model: Transformer 模型实例。
        tokenizer: 对应模型的 tokenizer。
        max_length (int): 最大序列长度。
        batch_size (int): 批大小。

    Returns:
        torch.Tensor: 所有序列的特征张量。
        numpy.ndarray: 所有序列的 attention_mask。
    """
    features = []
    attention_masks = []  # 用于存储每批次的 attention_mask

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        if pretrained_type == "gpt2":
            inputs = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, add_special_tokens=True)
        else:
            inputs = tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
	
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            if pretrained_type == "ProtT5":
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            else:
                outputs = model(**inputs)

        batch_features = outputs.last_hidden_state.cpu()
        features.append(batch_features)
        attention_mask = inputs['attention_mask'].cpu()
        attention_masks.append(attention_mask)

        del inputs, outputs
        torch.cuda.empty_cache()

    all_features = torch.cat(features, dim=0).numpy()
    all_attention_masks = torch.cat(attention_masks, dim=0).numpy()
    return all_features, all_attention_masks


def process_Cterminal_sequence(sequences, max_length):
    truncation_sequences = []
    for i, eachseq in enumerate(sequences):
        if len(eachseq) > max_length:
            truncation_sequences.append(eachseq[-max_length:])
        else:
            truncation_sequences.append(eachseq)
        # 4 debug
        print(f"序列 {i+1:03d}: 原始长度 = {len(eachseq)}, C-terminal 保留长度 = {len(eachseq[-max_length:])}")
        orint("real length\n")

    return truncation_sequences


def pool_features(features, attention_masks, pooling="mean"):
    if not torch.is_tensor(features):
        features = torch.tensor(features, dtype=torch.float32)
    if not torch.is_tensor(attention_masks):
        attention_masks = torch.tensor(attention_masks, dtype=torch.float32)

    attention_masks = attention_masks.unsqueeze(-1)

    if pooling == "mean":
        pooled = (features * attention_masks).sum(dim=1) / attention_masks.sum(dim=1)
    elif pooling == "max":
        pooled = (features * attention_masks).max(dim=1).values
    else:
        raise ValueError("Unsupported pooling type. Use 'mean' or 'max'.")
    return pooled


def compute_class_weights(labels):
    """
    计算类别的权重，依据阳性和阴性样本数量的反比
    :param labels: 真实标签（Tensor），1表示阳性，0表示阴性
    :return: 类别权重的Tensor
    """
    # 计算阳性和阴性样本数量
    positive_samples = (labels == 1).sum()
    negative_samples = (labels == 0).sum()

    # 计算权重，反比于样本数量
    total_samples = positive_samples + negative_samples
    weight_positive = total_samples / positive_samples
    weight_negative = total_samples / negative_samples

    # 返回权重，阳性权重大，阴性权重小
    return torch.tensor([weight_negative, weight_positive])


class WeightedSampler(Sampler):
    def __init__(self, data_list, class_weights):
        self.data_list = data_list
        self.class_weights = class_weights
        self.weights = self.compute_weights()

    def compute_weights(self):
        # 根据数据的标签（data.y）计算样本的权重
        weights = []
        for data in self.data_list:
            label = data.y.item()
            weight = self.class_weights[label]
            weights.append(weight)

        return torch.tensor(weights)

    def __iter__(self):
        # 返回一个迭代器，每次返回采样的样本索引
        return iter(torch.multinomial(self.weights, len(self.weights), replacement=True))

    def __len__(self):
        return len(self.data_list)


class WeightedSampler_4combine(Sampler):
    def __init__(self, labels, class_weights):
        """
        :param labels: 真实标签的列表或Tensor，长度应与数据集相同
        :param class_weights: 计算得到的类别权重
        """
        self.labels = labels
        self.class_weights = class_weights
        self.weights = self.compute_weights()

    def compute_weights(self):
        # 根据 labels 计算每个样本的权重
        weights = self.class_weights[self.labels]
        return weights

    def __iter__(self):
        # 返回一个迭代器，每次返回采样的样本索引
        return iter(torch.multinomial(self.weights, len(self.weights), replacement=True))

    def __len__(self):
        return len(self.labels)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def custom_collate(batch):
    feature_data = [item[0] for item in batch]  # 提取特征数据
    graph_data = [Data(x=item[1], edge_index=item[2], edge_attr=item[3], batch=item[4]) for item in batch]  # 提取图数据
    labels = [item[5] for item in batch]  # 提取标签

    batch_graph_data = Batch.from_data_list(graph_data)
    feature_tensor = torch.stack(feature_data, dim=0)

    return feature_tensor, batch_graph_data, torch.tensor(labels)


def load_model_without_dataparallel(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith('module.') else k
        new_state_dict[new_key] = v
    # model.load_state_dict(new_state_dict)
    model.load_state_dict(new_state_dict, strict=False)
    return model


def load_val_numpy(data_dir, filename):
    x_val = np.load(data_dir + filename, allow_pickle=True)['data']
    x_val = torch.from_numpy(x_val).float()
    y_val = np.load(data_dir + filename, allow_pickle=True)['labels']
    y_val = torch.from_numpy(y_val).long()
    return x_val, y_val 

def load_test_numpy(data_dir, pos_filename, neg_filename, pool=False):     
    pos_x = np.load(data_dir + pos_filename, allow_pickle=True)
    neg_x = np.load(data_dir + neg_filename, allow_pickle=True)
    features_raw = np.vstack((pos_x['embedding'], neg_x['embedding']))
    if pool==True:
         attention_masks_raw = np.vstack((pos_x['attention_masks'], neg_x['attention_masks']))
         x_test = pool_features(features_raw, attention_masks_raw, pooling="mean").numpy()
    else:
        x_test = features_raw
    x_test = torch.from_numpy(x_test).float()
    y_test = np.array([1] * pos_x['embedding'].shape[0] + [0] * neg_x['embedding'].shape[0])
    y_test = torch.from_numpy(y_test).long()
    del pos_x, neg_x, features_raw
    gc.collect()
    return x_test, y_test


def load_test_numpy_nopool(data_dir, pos_filename, neg_filename):
    pos_x = np.load(data_dir + pos_filename, allow_pickle=True)
    neg_x = np.load(data_dir + neg_filename, allow_pickle=True)
    features = np.vstack((pos_x['embedding'], neg_x['embedding']))
    attention = np.vstack((pos_x['attention_masks'], neg_x['attention_masks']))
    y_test = np.array([1] * pos_x['embedding'].shape[0] + [0] * neg_x['embedding'].shape[0])
    y_test = torch.from_numpy(y_test).long()
    return features, attention, y_test


def load_predict_numpy_nopool(data_dir, filename):
    data_x = np.load(data_dir + filename, allow_pickle=True)
    return data_x['embedding'], data_x['attention_masks'], data_x['seq_ids']
