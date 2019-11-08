# SEPT: Span Extractor with Pre-trained Transformer

This repo contains dataset and code for the article: **SEPT: Improving Scientific Named Entity Recognition with Span Representation**

# Before start

We preset a lot of configurations in `config` directory.
You can explore different model with different embeddings.
Please download the corresponding model parameters or embeddings in advance.
Save them in `embeddings` directory.
Download the [SCIERC](http://nlp.cs.washington.edu/sciIE/) dataset and save it to `data` directory.

# Training

```bash
python train.py -c config/sept.json -d [gpu]
```

