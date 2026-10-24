# BWIR672 Project: Neural Network for GPD Parameter Estimation

Coursework project for module **BWIR672** at North-West University. A PyTorch multilayer perceptron is trained to predict two parameters, `xi` and `sig`, from eleven numeric features (`Q1`–`Q11`). The file and column names (`gpd_f.csv`, `xi`, `sig`, `xi_mle`, `sig_mle`) indicate that the targets are the shape (ξ) and scale (σ) parameters of a Generalized Pareto Distribution (GPD), and that the dataset also has columns for their maximum-likelihood estimates.

## What it does

- **`explore.py`** gives a quick look at the dataset. It prints the column types, the first and last rows, the partition values in `_PartInd_` and their counts, and the size of the training partition after it keeps only the required columns and drops missing values.
- **`NN.py`** trains the network, keeps the weights from the epoch with the lowest validation loss in `best_model.pth`, and writes predictions for every row to `gdp_score.csv`.

## Data

Both scripts read **`gpd_f.csv`** from the working directory. The file is **not in this repository** because `.gitignore` excludes `*.csv`, so you need your own copy. It must contain these columns:

| Column | Role |
|---|---|
| `_PartInd_` | Partition indicator (SAS convention): `1` = training, `0` = validation, `2` = test |
| `Q1` … `Q11` | Input features, used unscaled |
| `xi`, `sig` | Regression targets |
| `N`, `iqr`, `xi_mle`, `sig_mle` | Required by the column check and carried through to the output, but not used for training |

Before dropping rows with missing values, `NN.py` sets `xi_mle` and `sig_mle` to `1` for the training and validation rows. This stops those rows from being dropped for missing MLE values.

## Model and training (`NN.py`)

- **Architecture:** a fully connected network, 11 → 512 → 512 → 512 → 128 → 128 → 2, with ReLU activations and a linear output layer (column 0 = `xi`, column 1 = `sig`). A dropout layer follows each hidden layer, with `DROPOUT_P = 0.0` by default.
- **Loss and optimiser:** MSE on both targets jointly. Adam with learning rate `1e-3`, betas `(0.9, 0.999)` and weight decay `0.0`.
- **Training:** batch size 128 (256 for validation), 60 epochs, gradient-norm clipping at 5.0, seed 42, CPU only.
- **Monitoring:** each epoch prints the training loss, validation loss, and validation RMSE and MAE.
- **Checkpointing:** whenever the validation loss improves, `model.state_dict()` is saved to `best_model.pth`. Those best weights are reloaded before the script makes predictions.

All hyperparameters are constants at the top of `NN.py`: `BATCH_SIZE`, `EPOCHS`, `LR`, `WEIGHT_DECAY`, `DROPOUT_P` and `SEED`.

## `best_model.pth`

This is a PyTorch `state_dict` for the `Net` class in `NN.py`: about 614k float32 parameters, 2.5 MB. It holds the weights from the best validation epoch of a training run. **Running `NN.py` overwrites it.** To use the weights without retraining, copy the `Net` class out of `NN.py` (importing `NN.py` would rerun training) and load them:

```python
import torch
# Net = <class copied from NN.py>
model = Net(in_features=11, out_features=2)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()
# inputs: float32 tensor of Q1..Q11 in that order, unscaled -> outputs [xi, sig]
```

## Project structure

```
explore.py        # inspect gpd_f.csv: dtypes, partitions, training-set size
NN.py             # train the MLP, save the best checkpoint, write gdp_score.csv
best_model.pth    # saved state_dict from the best validation epoch
requirements.txt  # Python dependencies
```

## Getting started

Prerequisites: Python 3 and `gpd_f.csv` in the repository root.

```bash
pip install -r requirements.txt
python explore.py   # optional: inspect the data
python NN.py        # train, checkpoint and score
```

`NN.py` produces:

- a per-epoch training log in the console
- `best_model.pth`, which is overwritten
- `gdp_score.csv`, the filtered dataset plus `pred_xi` and `pred_sig` for every row, including the test partition (`_PartInd_ == 2`)

## Notes

- The dataset is not included (see [Data](#data)).
- Inputs are deliberately not standardised.
- The test partition is split off but not scored inside the script. Its predictions are included in `gdp_score.csv`.
- A code comment says "tanh hidden", but the network uses ReLU. This README describes the code as written.
- Some comments mention an Excel export (`predictions_with_data.xlsx`), but the script only writes `gdp_score.csv`.
