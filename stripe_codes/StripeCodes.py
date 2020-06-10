#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:19:39 2020

@author: tvanzyl
"""

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


def edit_dist_even(C1,L1,C2,L2, len1, len2):
    #GENERATE ODD RANGE TO EVALUATE PAIRS
    min_path = 10000.0
    for base in (0,1):
        for c in combinations( range(base,len1-1,2), round((len1-len2)/2) ):
            # print(c)
            if len(c) > 0:
                # LE = L1.copy()
                # for i, p in enumerate(c):
                #     LE.pop(p-2*i)
                #     LE.pop(p-2*i)
                res = len(c)*0.6*2.0
            else:
                # LE = L1
                res = 0.0
            # print(L1, len(C1))
            # print(LE, len(CE))
            # print(L2, len(C2))
            j = base
            for i in range(base,len1,1):
                if i in c or i-1 in c:
                    #skip
                    ...
                else:
                    if L1[i] < L2[j]:
                        res += 1 - L1[i]/L2[j]
                    else:
                        res += 1 - L2[j]/L1[i]

                    # if L1[1+i] < L2[1+j]:
                    #     res += 1 - L1[1+i]/L2[1+j]
                    # else:
                    #     res += 1 - L2[1+j]/L1[1+i]
                    j += 1
            if res < min_path:
                min_path = res
    return min_path


def edit_dist(C1,L1,C2,L2):
    Xlen = len(C1)
    Ylen = len(C2)
    S1, T1, len1 = (C1, L1, Xlen) if Xlen  > Ylen else (C2, L2, Ylen)
    S2, T2, len2 = (C1, L1, Xlen) if Xlen <= Ylen else (C2, L2, Ylen)
    path = 0

    if len1 > 40:
        print(len1-len2, len1, len2)
        return None

    if len1 == len2:
        if S1[0] != S2[0]:
            #MISALIGNED
            path += 0.6*2
            #delete first item from fist string, recurse
            path1 = edit_dist_even(S1[1:],T1[1:],S2[:-1],T2[:-1],len1-1,len2-1)
            #delete first item from second string, recurse
            path2 = edit_dist_even(S1[:-1],T1[:-1],S2[1:],T2[1:],len1-1,len2-1)
            path += path1 if path1>path2 else path2
        else:
            #ALIGNED
            path += edit_dist_even(S1,T1,S2,T2,len1,len2)
    elif (len1-len2)%2 == 0:
        #EVEN CASE
        if S1[0] != S2[0]:
            #MISALIGNED
            #delete fist and last item from longest string
            path += 0.6*2.0
            path += edit_dist_even(S1[1:-1],T1[1:-1],S2,T2,len1-2,len2)
        else:
            #ALIGNED
            # print('S1',S1)
            # print('S2',S2)
            path += edit_dist_even(S1,T1,S2,T2,len1,len2)
    else:
        #ODD CASE
        if S1[0] != S2[0]:
            #MISALIGNED
            #delete first item of longest string and do even aligned combinations
            path += 0.6
            path += edit_dist_even(S1[1:],T1[1:],S2,T2,len1-1,len2)
        else:
            #ALIGNED
            #delete last item of longest string and do even aligned combinations
            path += 0.6
            path += edit_dist_even(S1[:-1],T1[:-1],S2,T2,len1-1,len2)
    return path/len1


def mean_edit_dist(C1,L1,C2,L2):
    sum_i = 0
    for i in range(10):
        # print('C1',C1[i][1:])
        # print('C2',C2[i][1:])
        tmp_i = edit_dist(C1[i][1:], L1[i][1:], C2[i][1:], L2[i][1:])
        if tmp_i is not None:
            sum_i += tmp_i/10
    return sum_i

dfc = pd.DataFrame(animalsc)
dfc = dfc.unstack().dropna().reset_index()
dfc = dfc.drop(columns=['level_1'])

dfl = pd.DataFrame(animalsl)
dfl = dfl.unstack().dropna().reset_index()
dfl = dfl.drop(columns=['level_1'])

# C1 = dfc.iloc[0,1]
# L1 = dfl.iloc[0,1]
# C2 = dfc.iloc[141,1]
# L2 = dfl.iloc[141,1]

# print(mean_edit_dist(C1,L1,C2,L2))

df = pd.DataFrame(columns=['a1','c1','a2','c2','dist'])

from joblib import Parallel, delayed

c = 0
for i in range(len(dfc)):
    for j in range(len(dfc)):
        C1 = dfc.iloc[i,1]
        L1 = dfl.iloc[i,1]
        C2 = dfc.iloc[j,1]
        L2 = dfl.iloc[j,1]
        dist = mean_edit_dist(C1,L1,C2,L2)
        df[c] = [dfc.iloc[i].level_0,i,dfc.iloc[j].level_0,j,dist]
        print(dfc.iloc[i].level_0,i,dfc.iloc[j].level_0,j,dist)
        c += 1








