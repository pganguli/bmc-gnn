#!/usr/bin/bash
set -euo pipefail

usage () {
  echo "Unfold input circuit to n depths, producing a row-sum'd embedding vector from DeepGate2 for every depth.

Usage:
$(basename "$0") [-h] -d n -c <input_circuit> -o <output_path>

where:
    -h      Print this message and exit.
    -d n    Depth upto which input circuit will be finally unrolled to.
    -c <input_circuit>
            Input circuit file in AIG format.
    -o <output_path>
            Path where unrolled circuits will be stored." >&2
}

while getopts ':hd:c:o:' option; do
  case "$option" in
    h) usage; exit;;
    d) DEPTH="$OPTARG";;
    c) CIRCUIT_IN="$OPTARG";;
    o) OUTPUT_PATH="$OPTARG";;
    :) echo "missing argument for -$OPTARG" >&2; usage; exit 1;;
   \?) echo "unknown option: -$OPTARG" >&2; usage; exit 1;;
    *) echo "unimplemented option: -$option" >&2; usage; exit 1;;
  esac
done
shift $((OPTIND - 1))

if [ -z "${DEPTH+x}" ] || [ -z "${CIRCUIT_IN+x}" ] || [ -z "${OUTPUT_PATH+x}" ]; then
  echo "One or more mandatory options missing." >&2; usage; exit 1
fi

for depth in $(seq 2 "${DEPTH}"); do
  CIRCUIT_OUT="${OUTPUT_PATH}/$(basename ${CIRCUIT_IN%.aig})_$(printf '%08d' ${depth}).aig"
  abc -c "read_aiger ${CIRCUIT_IN}; &get; &frames -F ${depth} -s -b; &write ${CIRCUIT_OUT}" && \
  ./rowsum_embedding.py -c "${CIRCUIT_OUT}" -o "${CIRCUIT_OUT%.aig}.pkl" && \
  rm "${CIRCUIT_OUT}"
done
