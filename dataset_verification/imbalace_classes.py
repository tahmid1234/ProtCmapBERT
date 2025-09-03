import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))
from helper.sample_stat import load_GO_annot, load_EC_annot
def analyse_class_imbalance(class_sizes,mean):

    pos_weights = mean /(class_sizes ) +1e-18
    print(f' beofre clip max = {np.max(pos_weights)} min ={np.min(pos_weights)}')
    sorted_pos_wights = pos_weights.copy()
    sorted_pos_wights.sort()
    print("mid id,", sorted_pos_wights[len(class_sizes)//2])


    max_val = np.sum(pos_weights<6)
    min_val = np.sum(pos_weights<3)
    pos_weights = np.clip(pos_weights,5.0,10.0)
    print(f' after clip max = {np.max(pos_weights)} min ={np.min(pos_weights)}')
    print(max_val," max and min value " ,min_val)
    print(len(class_sizes))
    

    
ontology = ['mf','bp','cc']
for ont in ontology:
        print(ont)
        prot2annot, goterms, gonames, counts = load_GO_annot("/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-GO_2019.06.18_annot.tsv")
        class_sizes = counts[ont]
        mean_class = np.mean(class_sizes) 
        print(len(prot2annot))
        print("mean clssa",mean_class)
print("EC")
prot2annot, goterms, gonames, counts = load_EC_annot("/mmfs1/home/mdtahmid.islam/deepfri_bert/preprocessing/data/nrPDB-EC_2020.04_annot.tsv")
class_sizes = counts["ec"]
mean_class = np.mean(class_sizes) 
print(len(prot2annot))
print("mean clssa",mean_class)
analyse_class_imbalance(class_sizes,mean_class)

