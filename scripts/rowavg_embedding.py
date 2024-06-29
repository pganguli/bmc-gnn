#!/usr/bin/env python3
"""DeepGate2 row aggregate of embedding tensor.

Produces a 128 sized vector with float values by taking an
aggregate on rows of a g x 128 tensor obtained from DeepGate2,
where g is the number of gates in the input circuit.

"""

import argparse
import pickle
from pathlib import Path

import deepgate
import torch


def produce_embedding(circuit_file: Path) -> torch.Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = deepgate.Model()
    model.load_pretrained()
    model = model.to(device)

    parser = deepgate.AigParser()
    graph = parser.read_aiger(circuit_file)
    graph = graph.to(device)

    hs, _ = model(graph)
    hs_rowavg = torch.mean(hs, dim=0)

    return hs_rowavg


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rowavg_embedding.py", description="DeepGate2 rowavg of embedding tensor."
    )
    parser.add_argument(
        "-c",
        "--circuit-file",
        help="Path to AIG circuit file.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output-pkl-file",
        help="Path to pickle file to save 128 sized row-avg'd embedding vector.",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    hs_rowavg = produce_embedding(args.circuit_file)

    with open(args.output_pkl_file, "wb") as pkl_file:
        pickle.dump(hs_rowavg, pkl_file)


if __name__ == "__main__":
    main()
