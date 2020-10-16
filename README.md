# SciNER: A Novel Scientific Named Entity Recognizing Framework

This supplementary material contains code for the article: **SciNER: A Novel Scientific Named Entity Recognizing Framework**

# Before start

We preset a lot of configurations containing the hyper-parameters we used in `config` directory.

Please download the corresponding model parameters or embeddings in advance. Save them in `embeddings` directory.

Download the [SCIERC](http://nlp.cs.washington.edu/sciIE/) dataset and save it to `data/scierc` directory.

For other datasets, we use as the same as [SciBERT](https://github.com/allenai/scibert).

For example: 

```
.
├── embeddings
│   └── scibert_scivocab_uncased
│       ├── vocab.txt
│       ├── config.json
│       └── pytorch_model.bin
├── data
│   └── scierc
│       ├── dev.json
│       ├── test.json
│       └── train.json
...
```

Requirements are listed in `requirements.txt`.


# Training

```bash
python train.py -c config/sept.json -d <gpu>
```

## Test

```bash
python test.py -r <saved_checkpoint_path> -d <gpu>
```


# Citing

If you use SciNER in your research, please cite our paper:

```
@InProceedings{10.1007/978-3-030-60450-9_65,
author="Yan, Tan
and Huang, Heyan
and Mao, Xian-Ling",
editor="Zhu, Xiaodan
and Zhang, Min
and Hong, Yu
and He, Ruifang",
title="SciNER: A Novel Scientific Named Entity Recognizing Framework",
booktitle="Natural Language Processing and Chinese Computing",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="828--839",
isbn="978-3-030-60450-9"
}
```
