#!/usr/bin/env python3

import sys

import deepgate
import torch

if __name__ == "__main__":
    model = deepgate.Model()
    model.load_pretrained()
    parser = deepgate.AigParser()

    aig_path: str = sys.argv[1]

    graph = parser.read_aiger(aig_path)
    hs, _ = model(graph)
    hs_rowsum = torch.sum(hs, dim=0)

    print(hs_rowsum)
