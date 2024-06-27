#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/latch_counts.tsv", sep="\t")
df_2000 = df[df["latches"] <= 2000]
counts, boundaries, _ = plt.hist(df_2000["latches"], bins=5)
