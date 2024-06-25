#!/usr/bin/bash

set -euo pipefail

echo 'F,O,And,Var,Conf,Cla,Learn,Memory,Time'

cat "${1}" | \
  tail -n +6 | \
  head -n -3 | \
  sed 's/.*://' | \
  sed 's/[^0-9\ \.]*//g' | \
  sed 's/\ $//' | \
  sed 's/^\ \+//' | \
  sed 's/\([0-9]\)\.\ /\1/g' | \
  sed 's/\ \+/,/g'
