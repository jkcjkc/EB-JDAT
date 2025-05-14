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


## Training

To train an EB-JDAT model on any JEM variant:

```
./train.sh
```


## Evaluation

All evaluation routines are consolidated in the  `<span>eval.py</span>` file. After training finishes, evaluate your model by running:

```
python eval.py
```
