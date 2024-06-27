#!/usr/bin/make -f

ABC ?= abc
CIRCUIT ?= ../data/chosen_circuits/6s7.aig
MAX_DEPTH ?= 10
PREFIX ?= /tmp

PICKLES := $(foreach n, $(shell seq -w 1 $(MAX_DEPTH)), ${PREFIX}/$(notdir $(basename ${CIRCUIT}))_$(n).pkl)

all: ${PICKLES}
	
%.pkl: %.aig
	poetry run ./rowavg_embedding.py -c "$<" -o "$@" && rm "$<"

%.aig:
	./unroll_circuit.sh "${CIRCUIT}" "$@"

clean:
	-rm -f ${PICKLES}

.INTERMEDIATE: %.aig
.PHONY: all clean
