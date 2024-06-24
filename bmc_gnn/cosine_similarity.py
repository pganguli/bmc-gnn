import time
import warnings

import deepgate
import torch

warnings.filterwarnings("ignore")
import aiger


def cosine_similarity(t1, t2):
    g1 = t1.size()[0]
    g2 = t2.size()[0]
    cos = torch.nn.CosineSimilarity(dim=0)
    max_cos = 0
    s = time.time()
    for i in range(g1):
        for j in range(g2):
            temp = cos(t1[i], t2[j])
            if temp > max_cos:
                max_cos = temp
    # print(f"For loop ended {time.time()-s} t1 size {t1.size()} t2 size {t2.size()}")
    return max_cos


def cosine_compare(circuit_1, circuit_2):
    start_time = time.time()
    model = deepgate.Model()
    # print(f"model loaded {round(time.time()-start_time,2)}")
    start_time = time.time()
    model.load_pretrained()
    # print(f"load_pretrained {round(time.time()-start_time,2)}")
    start_time = time.time()
    parser = deepgate.AigParser()
    # print(f"parser loaded {round(time.time()-start_time,2)}")
    start_time = time.time()
    graph_1 = parser.read_aiger(circuit_1)
    # print(f"graph_1 done {round(time.time()-start_time,2)}")
    start_time = time.time()
    graph_2 = parser.read_aiger(circuit_2)
    # print(f"graph_2 done {round(time.time()-start_time,2)}")
    start_time = time.time()
    hs_1, hf_1 = model(graph_1)
    # print(f"hs_1 and hf_1  done {round(time.time()-start_time,2)}")
    start_time = time.time()
    hs_2, hf_2 = model(graph_2)
    # print(f"hs_2 and hf_2  done {round(time.time()-start_time,2)}")
    start_time = time.time()
    cos_hs = cosine_similarity(hs_1, hs_2)
    # print(f"cosine sim calculation ended {round(time.time()-start_time,2)} hs_1 : {hs_1.size()} hs_2 : {hs_2.size()} circuit_1 : {circuit_1} circuit_2 : {circuit_2}")
    op = [circuit_2, cos_hs]
    return op
