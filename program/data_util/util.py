import numpy as np

import pandas as pd
from collections import OrderedDict

label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

def dataset_cnt_dict(dataset,data_number):
    class_with_cnt={}
    for i in range(data_number):
        keys = np.array(dataset.drop(['Id'],axis=1).Target)[i].split(" ")
        for key in keys:
            key = int(key)
            if key not in class_with_cnt.keys():
                class_with_cnt[key]=0
            class_with_cnt[key]+=1 
    return OrderedDict(sorted(class_with_cnt.items(), key=lambda t: t[1],reverse=True))


def get_class_weighted(class_dict,class_num):
    total_labels = np.sum(list(class_dict.values()))
    weighted_dict = {k:1.0/(v/total_labels) for k,v in class_dict.items()}
    class_weighted = []
    for i in range(class_num):
        class_weighted.append(weighted_dict[i])
    return class_weighted

def convert_class_to_name(dataset,data_number):
    class_with_cnt_dict = dataset_cnt_dict(dataset,data_number)
    class_to_name_dict = {label_names[k]:v for k,v in class_with_cnt_dict.items()}
    return OrderedDict(sorted(class_to_name_dict.items(), key=lambda t: t[1],reverse=True))




def get_data_info(path):
    data_info = pd.read_csv(path)
    dataset_total_num = data_info.shape[0]
    print('We have {} train data'.format(dataset_total_num))
    return data_info,dataset_total_num