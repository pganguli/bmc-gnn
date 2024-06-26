#!/usr/bin/bash

set -euo pipefail

DEPTH="${1}"
CIRCUIT_IN="${2}"
OUTPUT_PATH="${3%/}"

for depth in $(seq 2 "${DEPTH}"); do
  CIRCUIT_OUT="${OUTPUT_PATH}/$(basename ${CIRCUIT_IN%.aig})_$(printf '%08d' ${depth}).aig"
  abc -c "read_aiger ${CIRCUIT_IN}; &get; &frames -F ${depth} -s -b; &write ${CIRCUIT_OUT}"
  ./rowsum_embedding.py "${CIRCUIT_OUT}"
done
