# PyTorch Implementation of TESAN

This repo contains the code for our WWW 2020 paper: [An End-to-end Topic-Enhanced Self-Attention Network for Social Emotion Classification](https://dl.acm.org/doi/10.1145/3366423.3380286). Some code is based on [prodLDA](https://github.com/hyqneuron/pytorch-avitm). Thanks for sharing the code.

# Requirements

- python 3.5
- pytorch 0.3.1
- numpy
- pickle

# File Discription

- `load_data.py`: loading the dataset
- `parameter.py`: contains the parameters of TESAN
- `model_tesan.py`: the code of the TESAN model
- `tesan.py`: the code for training and testing TESAN
- `dataset/`
  - `SemEval_wo_swltf.txt`: SemEval 2007 Task 14 dataset. The stopwords and low frequency words are removed.
  - `data_bow_ltf.npy`: The bag-of-words(BOW) vector of each sample in the dataset.
  - `label.npy`: The label of each sample.
  - `word2vec.pickle`: The pre-trained word2vec model.

# Cite
  
If you find the code helpful, please kindly cite the paper:
```
@inproceedings{10.1145/3366423.3380286,

author = {Wang, Chang and Wang, Bang},

title = {An End-to-End Topic-Enhanced Self-Attention Network for Social Emotion Classification},

year = {2020},

publisher = {Association for Computing Machinery},

url = {https://doi.org/10.1145/3366423.3380286},

booktitle = {Proceedings of The Web Conference 2020},

pages = {2210–2219},

numpages = {10},

location = {Taipei, Taiwan},

series = {WWW '20}
}
```