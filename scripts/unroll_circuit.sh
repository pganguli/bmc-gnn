#!/usr/bin/bash
set -euo pipefail

DEPTH=$(echo "$2" | sed -e 's/\.aig$//; s/^.*_//; s/^0*//')
abc -c "read ${1}; &get; &frames -F ${DEPTH} -s -b; &write $2"
