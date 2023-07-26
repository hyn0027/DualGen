# AMR2Text Dual Encoder

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

