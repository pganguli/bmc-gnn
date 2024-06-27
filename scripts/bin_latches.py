#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/gate_counts.txt", sep=" ")
counts, boundaries, _ = plt.hist(df["M"], bins=5)
print(df[(df['M'] >= boundaries[0])  & (df['M'] < boundaries[1])])
print(df[(df['M'] >= boundaries[1])  & (df['M'] < boundaries[2])])
print(df[(df['M'] >= boundaries[2])  & (df['M'] < boundaries[3])])
print(df[(df['M'] >= boundaries[3])  & (df['M'] < boundaries[4])])
print(df[(df['M'] >= boundaries[4])  & (df['M'] < boundaries[5])])
plt.show()
