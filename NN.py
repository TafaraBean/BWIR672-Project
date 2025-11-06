import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

FEAT_COLS = [f"Q{i}" for i in range(1, 12)]
TGT_COLS  = ["xi", "sig"]
BATCH_SIZE = 128
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 0.0     # try 1e-4 if overfitting
DROPOUT_P = 0.0        # try 0.1 if overfitting
SEED = 42

# NEW: path to save the best model weights
BEST_MODEL_PATH = "best_model.pth"

torch.manual_seed(SEED)
np.random.seed(SEED)


device = torch.device("cpu")

# ----------------------------
# Load Excel (no standardization)
# ----------------------------
df = pd.read_csv("gpd_f.csv")

#let xi_mle and sig_mle be equal to 1 if in the training and validation set respectively
df.loc[df["_PartInd_"] == 1, "xi_mle"] = 1
df.loc[df["_PartInd_"] == 1, "sig_mle"] = 1
df.loc[df["_PartInd_"] == 0, "sig_mle"] = 1
df.loc[df["_PartInd_"] == 0, "xi_mle"] = 1


needed = ["_PartInd_","N"]  + FEAT_COLS + TGT_COLS + ["iqr","xi_mle","sig_mle"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in Excel: {missing}")

# Keep needed columns, drop NA, ensure numeric for features/targets
df = df[needed].dropna().copy()
for col in FEAT_COLS + TGT_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=FEAT_COLS + TGT_COLS)

train_df = df.loc[df["_PartInd_"] == 1].reset_index(drop=True)
val_df   = df.loc[df["_PartInd_"] == 0].reset_index(drop=True)
test_df  = df.loc[df["_PartInd_"] == 2].reset_index(drop=True)

if len(train_df) < 2 or len(val_df) < 1:
    raise ValueError("Not enough rows in train/validation after filtering. Check PartInd values.")

X_train = train_df[FEAT_COLS].to_numpy(dtype=np.float32)
Y_train = train_df[TGT_COLS].to_numpy(dtype=np.float32)
X_val   = val_df[FEAT_COLS].to_numpy(dtype=np.float32)
Y_val   = val_df[TGT_COLS].to_numpy(dtype=np.float32)

# ----------------------------
# Dataset / Dataloader
# ----------------------------
class XYDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_ds = XYDataset(X_train, Y_train)
val_ds   = XYDataset(X_val, Y_val)

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=(device.type == "cuda")
)
val_dl = DataLoader(
    val_ds, batch_size=256, shuffle=False,
    num_workers=0, pin_memory=(device.type == "cuda")
)

# ----------------------------
# Model: 11 -> 512,512,512,128,128 -> 2
# tanh hidden, linear output
# ----------------------------
class Net(nn.Module):
    def __init__(self, in_features=11, out_features=2, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, out_features)  # linear head
        )

    def forward(self, x):
        return self.net(x)

model = Net(in_features=11, out_features=2, p_drop=DROPOUT_P).to(device)

# ----------------------------
# Optimizer: Adam & MSE loss
# ----------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=WEIGHT_DECAY
)
criterion = nn.MSELoss()

def rmse(resid: torch.Tensor) -> float:
    return torch.sqrt((resid ** 2).mean()).item()

def mae(resid: torch.Tensor) -> float:
    return resid.abs().mean().item()

# ----------------------------
# Training loop + Save Best Checkpoint
# ----------------------------
best_val_loss = float("inf")  # NEW: track best validation loss

for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    train_losses = []

    for xb, yb in train_dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        yhat = model(xb)
        loss = criterion(yhat, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_losses.append(loss.item())

    # ---- validate ----
    model.eval()
    val_losses, rmses, maes = [], [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            val_losses.append(loss.item())
            resid = yhat - yb
            rmses.append(rmse(resid))
            maes.append(mae(resid))

    mean_train = float(np.mean(train_losses))
    mean_val = float(np.mean(val_losses))
    mean_rmse = float(np.mean(rmses))
    mean_mae = float(np.mean(maes))

    print(f"Epoch {epoch:3d} | "
          f"train {mean_train:.6f} | "
          f"val {mean_val:.6f} | "
          f"RMSE {mean_rmse:.6f} | "
          f"MAE {mean_mae:.6f}")

    # NEW: save best checkpoint when validation improves
    if mean_val < best_val_loss:
        best_val_loss = mean_val
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"âœ… Saved new best model (epoch {epoch}, val {best_val_loss:.6f})")

# NEW: load the best weights before inference/export
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()
print(f"âœ… Loaded best model from: {BEST_MODEL_PATH}")

# ----------------------------
# Inference (original scale) + optional CSV export
# ----------------------------
with torch.no_grad():
    Xv = torch.from_numpy(X_val).to(device)
    preds_val = model(Xv).cpu().numpy()

val_out = val_df.copy()
val_out["pred_xi"] = preds_val[:, 0]
val_out["pred_sig"] = preds_val[:, 1]

# Save predictions (uncomment to write)
# val_out.to_csv("val_predictions_python.csv", index=False)

print(val_out.head())

# ----------------------------
# Excel export with all rows + predictions
# ----------------------------
OUTPUT_XLSX = "predictions_with_data.xlsx"  # <- your output file

def predict_on_dataframe(model, df_part, feat_cols, device):
    """
    Returns numpy array of shape (n_rows, 2) with predictions [xi, sig].
    Assumes NO standardization (original scale).
    """
    if df_part.empty:
        return np.empty((0, 2), dtype=np.float32)

    X = df_part[feat_cols].to_numpy(dtype=np.float32, copy=True)
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        preds = model(X_t).cpu().numpy()
    return preds

# Predict for the entire dataframe (keeps original order) using BEST weights
preds_all = predict_on_dataframe(model, df, FEAT_COLS, device)

# Attach predictions to a copy of the original data
df_out = df.copy()
df_out["pred_xi"] = preds_all[:, 0]
df_out["pred_sig"] = preds_all[:, 1]



df_out.to_csv("gdp_score.csv", index=False)