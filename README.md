# DualGen

The code repo for [Two Heads Are Better Than One: Exploiting Both Sequence and Graph Models in AMR-To-Text Generation](https://openreview.net/forum?id=61DYdiyQqk).

## get silver data

```bash
cd pre-training
pip install -r requirements.txt
pip install amrlib
```

change the path in ``get_data.sh``


```bash
sh get_data.sh
```

silver data in ``silver_data/training/`` available

## preprocess

```bash
cd fairseq
```

change the path in preprocess.sh

```bash
sh preprocess.sh
```

## pre-training

```bash
cd fairseq
```

change the path in pre-train.sh

```bash
sh pre-train.sh
```

## fine-tune

```bash
cd fairseq
```

change the path in fine-tune.sh

```bash
sh fine-tune.sh
```

## train without pre-training

```bash
cd fairseq
```

change the path in train.sh

```bash
sh train.sh
```

