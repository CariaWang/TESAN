# coding: utf-8
import numpy as np

def load_data():
    # text
    text_data = []
    with open('dataset/SemEval_wo_swltf.txt', 'r', encoding='utf-8') as f:
        for line in f:
            text_data.append(line.strip('\n').split(' '))
    # label
    label = np.load('dataset/label.npy')

    return text_data, label