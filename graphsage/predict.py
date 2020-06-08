from graphsage.model import SupervisedGraphSage, load_cora
from graphsage.aggregators import MeanAggregator
from graphsage.encoders import Encoder

import torch
import torch.nn as nn
import numpy as np
import random

if __name__ == "__main__":
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    path = "/data/ducva/graphsage-simple/data/"

    cuda_available = False
    if torch.cuda.is_available():
        cuda_available = True
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if cuda_available:
        features.cuda()

    agg1 = MeanAggregator(features, cuda=cuda_available)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=cuda_available)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=cuda_available)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=cuda_available)
    model = SupervisedGraphSage(7, enc2)
    agg1.load_state_dict(torch.load(path+"MeanAggregator1.model", map_location=torch.device('cpu')))
    agg2.load_state_dict(torch.load(path+"MeanAggregator2.model", map_location=torch.device('cpu')))
    enc1.load_state_dict(torch.load(path+"Encoder1.model", map_location=torch.device('cpu')))
    enc2.load_state_dict(torch.load(path+"Encoder2.model", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(path+"SupervisedGraphSage.model", map_location=torch.device('cpu')))

    model.eval()
    paper = np.array([2203])
    cat = model.forward(paper)
    print(cat.data.numpy().argmax(axis=1))
