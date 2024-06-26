#!/usr/bin/env python3
"""DeepGate2 rowsum of embedding tensor.

Produces a 128 sized vector with float values by taking row-sum
of a g x 128 tensor obtained from DeepGate2, where g is the
number of gates in the input circuit.

"""

import argparse
import pickle
from pathlib import Path

import deepgate
import torch


def produce_embedding(circuit_file: Path) -> torch.Tensor:
    model = deepgate.Model()
    model.load_pretrained()
    parser = deepgate.AigParser()

    graph = parser.read_aiger(circuit_file)
    hs, _ = model(graph)
    hs_rowsum = torch.sum(hs, dim=0)

    return hs_rowsum


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rowsum_embedding.py", description="DeepGate2 rowsum of embedding tensor."
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
        help="Path to pickle file to save 128 sized row-sum'd embedding vector.",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    hs_rowsum = produce_embedding(args.circuit_file)

    with open(args.output_pkl_file, "wb") as pkl_file:
        pickle.dump(hs_rowsum, pkl_file)


if __name__ == "__main__":
    main()
