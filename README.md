# EB-JDAT

This repository provides the code for **EB-JDAT** implemented across different Joint Energy Models (JEM) variants. Each folder corresponds to a specific JEM architecture with ready-to-use training and evaluation scripts.

## Directory Structure

```
root/
├── EB-JDAT-JEM/
│   ├── train.sh         # Training script
│   ├── eval.py         # Test
│   └── ...
├── EB-JDAT-JEM++/
│   ├── train.sh         # Training script
│   ├── eval.py         # Test
│   └── ...
├── EB-JDAT-SADAJEM/
│   ├── train.sh         # Training script
│   ├── eval.py         # Test
│   └── ...
```


## Environment settings and libraries we used in our experiments

* OS: Ubuntu 20.04
* GPU: RTX 3090
* CUDA: 12.8
* Python: 3.9
* Libraries: see requirements.txt


## Training

To train an EB-JDAT model on any JEM variant:

```
./train.sh
```


## Evaluation

All evaluation routines are consolidated in the  *eval.py*  file. After training finishes, evaluate your model by running, including clf_test, PGD, AA, gen, FID and so on:

```
python eval.py
```
