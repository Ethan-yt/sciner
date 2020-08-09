# SciNER: A Novel Scientific Named Entity Recognizing Framework

This supplementary material contains dataset and code for the article: **SciNER: A Novel Scientific Named Entity Recognizing Framework**

# Before start

We preset a lot of configurations containing the hyper-parameters we used in `config` directory.

Please download the corresponding model parameters or embeddings in advance. Save them in `embeddings` directory.

Download the [SCIERC](http://nlp.cs.washington.edu/sciIE/) dataset and save it to `data/scierc` directory.

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

