# ============================================================
# QUATERNION-ONLY CONTACT + OVERLAP MODEL
# Predicting log10(overlap) directly (as in 00/01)
# + SIREN-style initialization for the Sine-activated network
# + separate contact/overlap loss logging
# + physics-informed slope penalty: d(overlap)/d(Z_POS) ~= -1
#   (CPU-only version, per user preference — no GPU/device code)
# ============================================================
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Explicitly request at least 16 CPU threads for PyTorch's intra-op
# parallelism (matrix multiplications, elementwise ops, etc.)
torch.set_num_threads(16)
torch.set_num_interop_threads(2)  # controls parallelism across independent ops; keep modest
print(f"PyTorch using {torch.get_num_threads()} intra-op threads")

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "training_data_sweep_v2.csv"   # swap to training_data_sweep_v2.csv once regenerated with jitter/gap-closing
MODEL_PATH = "overlap_model.pth"
TRACED_MODEL_PATH = "overlap_model.pt"
SCALER_PATH = "y_scaler.save"
SCALER_ASCII_PATH = "scalers.dat"

EPOCHS = 10500
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-7
PATIENCE = 1500

MIN_OVERLAP = 1e-6
EPS = 1e-15

CONTACT_THRESHOLD = 0.5
CUBE_SIZE = 1.0
HALF = CUBE_SIZE / 2.0

# ------------------------------------------------------------
# PHYSICS-INFORMED SLOPE PENALTY
# ------------------------------------------------------------
# The true relationship overlap = HALF*S - Z_POS is exactly affine
# in Z_POS for any fixed orientation, with slope exactly -1. This
# penalty term directly encodes that known geometry as a training
# signal, computed via autograd through the model (not finite
# differences), so it contributes properly to gradient updates.
# Motivated by the off-anchor slope check finding mean slope=-0.42
# (target -1.0) with std=48 on the previous sweep model — evidence
# the network was memorizing per-anchor behavior rather than
# learning the true global height dependence.
# ------------------------------------------------------------
LAMBDA_SLOPE = 0.05   # weight for the slope penalty term; tune if needed

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# LOAD DATA — columns read by name, not assumed by position
# ============================================================
df = pd.read_csv(DATASET_PATH)
print(f"\nAll samples: {len(df)}")
print(df.info())

QW_COL, QX_COL, QY_COL, QZ_COL = "Q_orientation.w()", "Q_orientation.x()", "Q_orientation.y()", "Q_orientation.z()"
POSZ_COL, OVERLAP_COL = "position_z", "overlap(m)"

position_z = df[POSZ_COL].values.astype(np.float32)
qw = df[QW_COL].values.astype(np.float32)
qx = df[QX_COL].values.astype(np.float32)
qy = df[QY_COL].values.astype(np.float32)
qz = df[QZ_COL].values.astype(np.float32)

# ------------------------------------------------------------
# Quaternion canonicalization (input only; q and -q represent
# the same orientation)
# ------------------------------------------------------------
quats = np.stack([qw, qx, qy, qz], axis=1)
mask = quats[:, 0] < 0.0
quats[mask] *= -1.0

X = np.column_stack([position_z, quats]).astype(np.float32)
print("\nInput shape:", X.shape)

# ============================================================
# TARGETS — log10(overlap)
# ============================================================
overlap = df[[OVERLAP_COL]].values.astype(np.float32)
contact = (overlap > 0.0).astype(np.float32)
overlap_safe = np.maximum(overlap, MIN_OVERLAP)
overlap_log = np.log10(overlap_safe + EPS)

y = np.hstack([contact, overlap_log]).astype(np.float32)
print("Target shape:", y.shape)

true_overlap_all = overlap.copy()

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
X_train_raw, X_test_raw, \
y_train_raw, y_test_raw, \
true_overlap_train, true_overlap_test = train_test_split(
    X, y, true_overlap_all,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ============================================================
# NORMALIZATION
# ------------------------------------------------------------
# IMPORTANT: only overlap_log is scaled; contact label remains
# untouched
# ============================================================
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train_raw)
X_test = x_scaler.transform(X_test_raw)

overlap_scaler = StandardScaler()
overlap_train_scaled = overlap_scaler.fit_transform(y_train_raw[:, 1:2])
overlap_test_scaled = overlap_scaler.transform(y_test_raw[:, 1:2])

y_train = np.hstack([y_train_raw[:, 0:1], overlap_train_scaled])
y_test = np.hstack([y_test_raw[:, 0:1], overlap_test_scaled])

# ============================================================
# TENSORS
# ============================================================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ============================================================
# DATALOADERS
# ============================================================
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# SINE ACTIVATION
# ============================================================
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# ============================================================
# MODEL (full-width architecture, matching 00)
# ============================================================
model = nn.Sequential(
    nn.Linear(5, 512),
    Sine(),
    nn.Linear(512, 256),
    Sine(),
    nn.Linear(256, 128),
    Sine(),
    nn.Linear(128, 64),
    Sine(),
    nn.Linear(64, 32),
    Sine(),
    nn.Linear(32, 16),
    Sine(),
    # --------------------------------------------
    # Output: [ contact_logit , overlap_log ]
    # --------------------------------------------
    nn.Linear(16, 2)
)

# ============================================================
# SIREN-STYLE INITIALIZATION
# ------------------------------------------------------------
# Standard PyTorch default init for nn.Linear is not well suited to networks with repeated sine activations (SIREN-style nets) -
# gradients through stacked sin() layers can vanish/alias with default init, causing the network to get stuck in a shallow
# local minimum early in training. This applies the initialization scheme from Sitzmann et al. (2020), "Implicit Neural
# Representations with Periodic Activation Functions".
# ============================================================
def siren_init(module, is_first_layer=False, omega_0=30.0):
    if isinstance(module, nn.Linear):
        fan_in = module.weight.shape[1]
        if is_first_layer:
            bound = 1.0 / fan_in
        else:
            bound = math.sqrt(6.0 / fan_in) / omega_0
        with torch.no_grad():
            module.weight.uniform_(-bound, bound)
            if module.bias is not None:
                module.bias.uniform_(-bound, bound)

linear_layers = [m for m in model if isinstance(m, nn.Linear)]
for i, layer in enumerate(linear_layers):
    siren_init(layer, is_first_layer=(i == 0))
print(f"\nApplied SIREN-style init to {len(linear_layers)} Linear layers.")

# ============================================================
# LOSSES
# ============================================================
bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()

# ------------------------------------------------------------
# PHYSICS-INFORMED SLOPE PENALTY FUNCTION
# ------------------------------------------------------------
# Computes d(overlap)/d(Z_POS) via autograd through the scaled
# input, converting the scaled log10(overlap) output back to raw
# overlap units via the chain rule, then penalizes squared
# deviation from the analytically known target slope of -1.
# Only applied on rows where contact_mask is True, mirroring the
# existing overlap-MSE masking pattern.
# ------------------------------------------------------------
def slope_penalty(model, xb, contact_mask, x_scaler, overlap_scaler, eps_val):
    xb_req = xb.detach().clone().requires_grad_(True)
    pred = model(xb_req)
    overlap_log_scaled = pred[:, 1:2]

    grad_out = torch.autograd.grad(
        outputs=overlap_log_scaled,
        inputs=xb_req,
        grad_outputs=torch.ones_like(overlap_log_scaled),
        create_graph=True,
        retain_graph=True,
    )[0]
    d_ylog_scaled_dz_scaled = grad_out[:, 0:1]  # column 0 = position_z

    y_scale = float(overlap_scaler.scale_[0])
    y_mean = float(overlap_scaler.mean_[0])
    x_scale_z = float(x_scaler.scale_[0])

    log10_overlap_raw = overlap_log_scaled * y_scale + y_mean
    overlap_val = torch.pow(10.0, log10_overlap_raw) - eps_val

    slope = (math.log(10.0) * (overlap_val + eps_val) * y_scale
             * d_ylog_scaled_dz_scaled / x_scale_z)

    if contact_mask.sum() > 0:
        return torch.mean((slope[contact_mask] - (-1.0)) ** 2)
    return torch.tensor(0.0, device=xb.device)

# ============================================================
# OPTIMIZER / SCHEDULER
# ============================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=200
)

# ============================================================
# TRAINING
# ------------------------------------------------------------
# The combined loss AND the three individual components
# (contact / overlap / slope) are tracked separately, per epoch,
# for both train and test sets - this lets us see directly which
# term is/isn't learning, instead of only seeing their sum.
# ============================================================
best_test_loss = np.inf
counter = 0

train_losses = []
test_losses = []
train_losses_contact = []
train_losses_overlap = []
train_losses_slope = []
test_losses_contact = []
test_losses_overlap = []
test_losses_slope = []

prev_lr = optimizer.param_groups[0]['lr']

for epoch in range(EPOCHS):
    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    model.train()
    train_loss_epoch = 0.0
    train_loss_contact_epoch = 0.0
    train_loss_overlap_epoch = 0.0
    train_loss_slope_epoch = 0.0

    for xb, yb in train_loader:
        pred = model(xb)
        contact_logits = pred[:, 0:1]
        overlap_pred = pred[:, 1:2]
        contact_true = yb[:, 0:1]
        overlap_true = yb[:, 1:2]

        loss_contact = bce_loss(contact_logits, contact_true)

        mask = contact_true > 0.5
        if mask.sum() > 0:
            loss_overlap = mse_loss(overlap_pred[mask], overlap_true[mask])
        else:
            loss_overlap = torch.tensor(0.0, device=xb.device)

        loss_slope = slope_penalty(model, xb, mask.squeeze(-1), x_scaler, overlap_scaler, EPS)

        loss = loss_contact + loss_overlap + LAMBDA_SLOPE * loss_slope

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item()
        train_loss_contact_epoch += loss_contact.item()
        train_loss_overlap_epoch += loss_overlap.item()
        train_loss_slope_epoch += loss_slope.item()

    train_loss_epoch /= len(train_loader)
    train_loss_contact_epoch /= len(train_loader)
    train_loss_overlap_epoch /= len(train_loader)
    train_loss_slope_epoch /= len(train_loader)

    # --------------------------------------------------------
    # TEST
    # ------------------------------------------------------------
    # Note: slope_penalty requires gradients through the input,
    # so it CANNOT run inside torch.no_grad(). It is computed here
    # with grad enabled for xb only (model weights still get no
    # optimizer step during test, since we never call .backward()
    # on the outer loss during this block).
    # ------------------------------------------------------------
    model.eval()
    test_loss_epoch = 0.0
    test_loss_contact_epoch = 0.0
    test_loss_overlap_epoch = 0.0
    test_loss_slope_epoch = 0.0

    for xb, yb in test_loader:
        with torch.no_grad():
            pred = model(xb)
            contact_logits = pred[:, 0:1]
            overlap_pred = pred[:, 1:2]
            contact_true = yb[:, 0:1]
            overlap_true = yb[:, 1:2]

            loss_contact = bce_loss(contact_logits, contact_true)

            mask = contact_true > 0.5
            if mask.sum() > 0:
                loss_overlap = mse_loss(overlap_pred[mask], overlap_true[mask])
            else:
                loss_overlap = torch.tensor(0.0, device=xb.device)

        # slope_penalty needs grad w.r.t. its own internal input clone,
        # so it's called outside the no_grad() block above
        loss_slope = slope_penalty(model, xb, mask.squeeze(-1), x_scaler, overlap_scaler, EPS)

        loss = loss_contact + loss_overlap + LAMBDA_SLOPE * loss_slope

        test_loss_epoch += loss.item()
        test_loss_contact_epoch += loss_contact.item()
        test_loss_overlap_epoch += loss_overlap.item()
        test_loss_slope_epoch += loss_slope.item()

    test_loss_epoch /= len(test_loader)
    test_loss_contact_epoch /= len(test_loader)
    test_loss_overlap_epoch /= len(test_loader)
    test_loss_slope_epoch /= len(test_loader)

    scheduler.step(test_loss_epoch)

    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != prev_lr:
        print(f"  >>> LR reduced at epoch {epoch}: {prev_lr:.2e} -> {new_lr:.2e}")
        prev_lr = new_lr

    train_losses.append(train_loss_epoch)
    test_losses.append(test_loss_epoch)
    train_losses_contact.append(train_loss_contact_epoch)
    train_losses_overlap.append(train_loss_overlap_epoch)
    train_losses_slope.append(train_loss_slope_epoch)
    test_losses_contact.append(test_loss_contact_epoch)
    test_losses_overlap.append(test_loss_overlap_epoch)
    test_losses_slope.append(test_loss_slope_epoch)

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------
    if test_loss_epoch < best_test_loss:
        best_test_loss = test_loss_epoch
        torch.save(model.state_dict(), MODEL_PATH)
        counter = 0
    else:
        counter += 1

    if counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

    if epoch % 100 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:5d} | "
            f"Train = {train_loss_epoch:.6e} (contact={train_loss_contact_epoch:.4e}, "
            f"overlap={train_loss_overlap_epoch:.4e}, slope={train_loss_slope_epoch:.4e}) | "
            f"Test = {test_loss_epoch:.6e} (contact={test_loss_contact_epoch:.4e}, "
            f"overlap={test_loss_overlap_epoch:.4e}, slope={test_loss_slope_epoch:.4e}) | "
            f"LR = {lr:.2e}"
        )

# ============================================================
# LOAD BEST MODEL
# ============================================================
model.load_state_dict(torch.load(MODEL_PATH))

# ============================================================
# FINAL EVALUATION
# ============================================================
model.eval()
with torch.no_grad():
    train_pred_all = model(X_train).numpy()
    test_pred_all = model(X_test).numpy()

# ============================================================
# CONTACT PREDICTION
# ============================================================
train_contact_prob = 1.0 / (1.0 + np.exp(-train_pred_all[:, 0:1]))
test_contact_prob = 1.0 / (1.0 + np.exp(-test_pred_all[:, 0:1]))
train_contact_pred = (train_contact_prob > CONTACT_THRESHOLD)
test_contact_pred = (test_contact_prob > CONTACT_THRESHOLD)

# ============================================================
# OVERLAP RECOVERY (from log10)
# ============================================================
train_overlap_scaled = train_pred_all[:, 1:2]
test_overlap_scaled = test_pred_all[:, 1:2]

train_overlap_log = overlap_scaler.inverse_transform(train_overlap_scaled)
test_overlap_log = overlap_scaler.inverse_transform(test_overlap_scaled)

train_pred_overlap = 10.0**train_overlap_log - EPS
test_pred_overlap = 10.0**test_overlap_log - EPS

train_pred_overlap[~train_contact_pred] = 0.0
test_pred_overlap[~test_contact_pred] = 0.0

# ============================================================
# METRICS
# ============================================================
test_rmse = np.sqrt(np.mean((test_pred_overlap - true_overlap_test) ** 2))
relative_rmse = test_rmse / np.mean(true_overlap_test)
test_contact_true = (y_test[:, 0:1].numpy() > 0.5)
contact_accuracy = np.mean(test_contact_true == test_contact_pred)

print("\n================================================")
print("RESULTS")
print("================================================")
print(f"Contact accuracy = {100*contact_accuracy:.2f} %")
print(f"Test RMSE = {test_rmse:.6e} m")
print(f"Relative RMSE = {100*relative_rmse:.4f} %")

# ============================================================
# LOSS CURVES (combined + separated components, incl. slope)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label="Train (combined)")
axes[0].plot(test_losses, label="Test (combined)")
axes[0].set_yscale("log")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Combined Loss")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(train_losses_contact, label="Train contact (BCE)")
axes[1].plot(test_losses_contact, label="Test contact (BCE)")
axes[1].plot(train_losses_overlap, label="Train overlap (MSE)")
axes[1].plot(test_losses_overlap, label="Test overlap (MSE)")
axes[1].plot(train_losses_slope, label="Train slope penalty", linestyle="--")
axes[1].plot(test_losses_slope, label="Test slope penalty", linestyle="--")
axes[1].set_yscale("log")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Separated Loss Components")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("loss_curve.png", dpi=300)
#plt.show()

# ============================================================
# PREDICTED VS TRUE (DEM-range)
# ============================================================
plt.figure(figsize=(7, 7))
plt.scatter(true_overlap_train, train_pred_overlap, s=8, alpha=0.4, label="Train")
plt.scatter(true_overlap_test, test_pred_overlap, s=8, alpha=0.4, label="Test")
plt.plot([1e-6, 1e-2], [1e-6, 1e-2], 'k--')
plt.xlabel("True overlap")
plt.ylabel("Predicted overlap")
plt.title("DEM-range: Predicted vs True (overlap prediction)")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-4, 1e-1)
plt.ylim(1e-4, 1e-1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pred_vs_true_DEM_range.png", dpi=300)
#plt.show()

# ============================================================
# SAVE MODEL / SCALERS
# ============================================================
torch.save(model.state_dict(), MODEL_PATH)
example = torch.randn(1, 5)
traced_model = torch.jit.trace(model, example, check_trace=True)
traced_model.save(TRACED_MODEL_PATH)

joblib.dump(x_scaler, "x_scaler.save")
joblib.dump(overlap_scaler, SCALER_PATH)

with open(SCALER_ASCII_PATH, "w") as f:
    f.write("x_mean\n")
    f.write(" ".join(map(str, x_scaler.mean_)) + "\n")
    f.write("x_scale\n")
    f.write(" ".join(map(str, x_scaler.scale_)) + "\n")
    f.write("y_mean\n")
    f.write(" ".join(map(str, overlap_scaler.mean_)) + "\n")
    f.write("y_scale\n")
    f.write(" ".join(map(str, overlap_scaler.scale_)) + "\n")
    f.write("eps\n")
    f.write(f"{EPS}\n")

print("\nModels and scalers saved.")
