#!/usr/bin/bash

set -euo pipefail

echo 'F,Var,Cla,Conf,Learn,Memory1,Memory2,Time'

cat "${1}" | \
  tail -n +6 | \
  head -n -5 | \
  sed 's/[^0-9 \.]*//g' | \
  sed 's/\ \+$//' | \
  sed 's/^\ \+//' | \
  sed 's/\([0-9]\)\.\ /\1/g' | \
  sed 's/\ \+/,/g' | \
  sed '/^3600,\.$/d'
