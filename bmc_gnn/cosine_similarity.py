import warnings

warnings.filterwarnings("ignore")

import aiger
import deepgate
import torch


def cosine_similarity(t1, t2):
    cos = torch.nn.CosineSimilarity(dim=0)
    g1 = t1.size()[0]
    g2 = t2.size()[0]
    max_cos = max([cos(t1[i], t2[j]) for i in range(g1) for j in range(g2)])
    return max_cos


def cosine_compare(circuit_1, circuit_2):
    model = deepgate.Model()
    model.load_pretrained()
    parser = deepgate.AigParser()
    graph_1 = parser.read_aiger(circuit_1)
    graph_2 = parser.read_aiger(circuit_2)
    hs_1, hf_1 = model(graph_1)
    hs_2, hf_2 = model(graph_2)
    cos_hs = cosine_similarity(hs_1, hs_2)
    op = [circuit_2, cos_hs]
    return op
