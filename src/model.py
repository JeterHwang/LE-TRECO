import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import OrderedDict
import math
import logging
import numpy as np

class Embedding(nn.Module):
    def __init__(
        self, 
        embed_type,
        num_layers, 
        model,
        alphabet=None
    ):
        super(Embedding, self).__init__()

        self.num_layers = num_layers
        self.embed_type = embed_type

        self.model = model
        self.alphabet = alphabet
        
    @staticmethod
    def load_pretrained(embed_type='prose_dlm', model_path=None):
        if embed_type == 'esm-43M':
            num_layers, hidden_dim = 6, 768
            model, esm_alphabet = pretrained.esm1_t6_43M_UR50S(model_path)
        elif embed_type == 'esm-35M':
            num_layers, hidden_dim = 12, 480
            model, esm_alphabet = pretrained.esm2_t12_35M_UR50D(model_path)
        elif embed_type == 'esm-150M':
            num_layers, hidden_dim = 30, 640
            model, esm_alphabet = pretrained.esm2_t30_150M_UR50D(model_path)
        elif embed_type == 'esm-650M':
            num_layers, hidden_dim = 33, 1280
            model, esm_alphabet = pretrained.esm2_t33_650M_UR50D(model_path)
        else:
            num_layers, hidden_dim = 3, 1024
            model, esm_alphabet = nn.LSTM(
                21,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=True
            ), None
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        
        model = Embedding(
            21, 
            hidden_dim, 
            num_layers, 
            model,
            esm_alphabet
        )
        
        return model

    def to_one_hot(self, x):
        packed = type(x) is PackedSequence
        if packed:
            one_hot = x.data.new(x.data.size(0), self.nin).float().zero_()
            one_hot.scatter_(1, x.data.unsqueeze(1), 1)
            one_hot = PackedSequence(one_hot, x.batch_sizes)
        else:
            one_hot = x.new(x.size(0), x.size(1), self.nin).float().zero_()
            one_hot.scatter_(2, x.unsqueeze(2), 1)
        return one_hot

    def forward(self, x, length):
        if 'esm' in self.embed_type:
            results = self.model(x, repr_layers=[self.repr_layers], return_contacts=False)
            hs = results["representations"][self.repr_layers][:,1:,:]
        else:
            one_hot = self.to_one_hot(x)
            h_ = pack_padded_sequence(one_hot, length, batch_first=True, enforce_sorted=False)
            h_, (hidden, cell) = self.model(h_)
            h_unpacked, _ = pad_packed_sequence(h_, batch_first=True)
            hs = h_unpacked

        pooling = []
        for i in range(x.size(0)):
            pooling.append(torch.mean(hs[i][:length[i]], dim=0))
        emb = torch.stack(pooling, dim=0)

        return emb