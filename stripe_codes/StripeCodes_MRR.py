#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 08:39:29 2020

@author: tvanzyl
"""

import pandas as pd


from itertools import combinations
import pandas as pd
from numba import jit

f_sc = open('StripeCodes.txt')
str_sc = f_sc.readlines()
f_sc.close()

animalsc = {}
animalsl = {}

# Animal loop
for line in str_sc:
    if line.startswith("ANIMAL"):
        animal = line.split()[1]
        code = line.split()[2]
        animalsc[animal] = animalsc.get(animal, {})
        animalsc[animal][code] = [[],[],[],[],[],[],[],[],[],[]]
        animalsl[animal] = animalsl.get(animal, {})
        animalsl[animal][code] = [[],[],[],[],[],[],[],[],[],[]]
    elif line.startswith("STRIPECODE"):
        code_length = int(line.split()[1])
    elif line.startswith("stripestring"):
        stripestring = int(line.split()[1])
        animalsc[animal][code][stripestring] = []
        animalsl[animal][code][stripestring] = []
    elif line.startswith("#"):
        animalsc[animal][code][stripestring].append( int(float(line.split()[1])/255) )
        animalsl[animal][code][stripestring].append( float(line.split()[-1]) )

dfc = pd.DataFrame(animalsc)
dfc = dfc.unstack().dropna().reset_index()
dfc = dfc.drop(columns=['level_1'])

dfl = pd.DataFrame(animalsl)
dfl = dfl.unstack().dropna().reset_index()
dfl = dfl.drop(columns=['level_1'])

df = pd.read_csv('./stripecode_dist.csv',)
df.rename(columns={'Unnamed: 0':'idx','0':'a1','1':'c1','2':'a2','3':'c2','4':'dist'},inplace=True)
df.drop(columns=['idx'],inplace=True)

import numpy as np

np.random.seed(0)
for k in (1,5,10,20,50,100):
    tot = 0
    for l in range(30):
        p_idx = np.random.permutation(pd.unique(df.c1))
        test_idx  = p_idx[:int(len(p_idx)*0.25)]
        train_idx = p_idx[int(len(p_idx)*0.25):]
        df_Q  = df.loc[df.c1.isin(test_idx) & df.c2.isin(train_idx)]
        for i in test_idx:
            Q = df_Q.loc[df_Q.c1 == i].sort_values('dist')[0:k]
            tot += np.any(Q.a1 == Q.a2)

    print("top-",k, " accuracy ", tot/len(test_idx)/30)

np.random.seed(0)
tot = 0
for l in range(30):
    p_idx = np.random.permutation(pd.unique(df.c1))
    test_idx  = p_idx[:int(len(p_idx)*0.25)]
    train_idx = p_idx[int(len(p_idx)*0.25):]
    df_Q  = df.loc[df.c1.isin(test_idx) & df.c2.isin(train_idx)]
    for i in test_idx:
        Q = df_Q.loc[df_Q.c1 == i].sort_values('dist')
        tot += 1/(np.argmin(Q.a1 != Q.a2)+1)

print("MRR ", tot/len(test_idx)/30)

np.random.seed(0)
for k in (1,5,10,20):
    tot = 0
    for l in range(100):
        p_idx = np.random.permutation(pd.unique(df.c1))
        test_idx  = p_idx[:int(len(p_idx)*0.25)]
        train_idx = p_idx[int(len(p_idx)*0.25):]
        df_Q  = df.loc[df.c1.isin(test_idx) & df.c2.isin(train_idx)]
        for i in test_idx:
            Q = df_Q.loc[df_Q.c1 == i][0:k]
            tot += np.any(Q.a1 == Q.a2)

    print("top-",k, " accuracy ", tot/len(test_idx)/100)


np.random.seed(0)
tot = 0
for l in range(100):
    p_idx = np.random.permutation(pd.unique(df.c1))
    test_idx  = p_idx[:int(len(p_idx)*0.25)]
    train_idx = p_idx[int(len(p_idx)*0.25):]
    df_Q  = df.loc[df.c1.isin(test_idx) & df.c2.isin(train_idx)]
    for i in test_idx:
        Q = df_Q.loc[df_Q.c1 == i]
        tot += 1/(np.argmin(Q.a1 != Q.a2)+1)

print("MRR ", tot/len(test_idx)/100)

s = 1945
ind = 237
r = 8.21

np.random.seed(0)
for k in (1,5,10,20,50,100):
    tot = 0
    for l in range(100):
        test_idx = np.random.permutation( np.random.choice(range(ind),1945)  )
        for i in test_idx:
            Q = test_idx[0:k]
            tot += np.isin(i, Q)

    print("top-",k, " accuracy ", tot/len(test_idx)/100)


np.random.seed(0)
tot = 0
for l in range(100):
    test_idx = np.random.permutation( np.random.choice(range(ind),1945)  )
    for i in test_idx:
        Q = test_idx
        tot += 1/(np.argmin(Q != i)+1)

print("MRR ", tot/len(test_idx)/100)



















