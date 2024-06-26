#!/usr/bin/make -f

TIMELIMIT := 2
FLAGS := -S 0 -T ${TIMELIMIT} -F 0 -v

ABC := ../abc
PREFIX := .

SRCS := $(wildcard *.aig)
OBJS := ${SRCS:.aig=.txt}

.PHONY: all clean
all: bmc2 bmc3 bmc3r bmc3s bmc3g bmc3u bmc3j
clean:
	rm -f bmc2/*.txt
	rm -f bmc3/*.txt
	rm -f bmc3r/*.txt
	rm -f bmc3s/*.txt
	rm -f bmc3g/*.txt
	rm -f bmc3u/*.txt
	rm -f bmc3j/*.txt

bmc2: $(foreach obj,$(OBJS),${PREFIX}/bmc2/$(obj))
bmc3: $(foreach obj,$(OBJS),${PREFIX}/bmc3/$(obj))
bmc3r: $(foreach obj,$(OBJS),${PREFIX}/bmc3r/$(obj))
bmc3s: $(foreach obj,$(OBJS),${PREFIX}/bmc3s/$(obj))
bmc3g: $(foreach obj,$(OBJS),${PREFIX}/bmc3g/$(obj))
bmc3u: $(foreach obj,$(OBJS),${PREFIX}/bmc3u/$(obj))
bmc3j: $(foreach obj,$(OBJS),${PREFIX}/bmc3j/$(obj))

${PREFIX}/bmc2/%.txt: %.aig
	${ABC} -c "read $<; print_stats; &get; bmc2 ${FLAGS} -L $@; print_stats"
${PREFIX}/bmc3/%.txt: %.aig
	${ABC} -c "read $<; print_stats; &get; bmc3 ${FLAGS} -L $@; print_stats"
${PREFIX}/bmc3r/%.txt: %.aig
	${ABC} -c "read $<; print_stats; &get; bmc3 -r ${FLAGS} -L $@; print_stats"
${PREFIX}/bmc3s/%.txt: %.aig
	${ABC} -c "read $<; print_stats; &get; bmc3 -s ${FLAGS} -L $@; print_stats"
${PREFIX}/bmc3g/%.txt: %.aig
	${ABC} -c "read $<; print_stats; &get; bmc3 -g ${FLAGS} -L $@; print_stats"
${PREFIX}/bmc3u/%.txt: %.aig
	${ABC} -c "read $<; print_stats; &get; bmc3 -u ${FLAGS} -L $@; print_stats"
${PREFIX}/bmc3j/%.txt: %.aig
	${ABC} -c "read $<; print_stats; &get; bmc3 ${FLAGS} -J 2 -L $@; print_stats"
