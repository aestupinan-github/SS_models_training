import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


EPOCHS=5000
PATIENCE=500

# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("training_data.csv")

print(f"\nAll samples: {len(df)}")
print(df.info())

# ============================================================
# REMOVE ZERO OVERLAPS
# ============================================================

min_overlap = 1e-14

df = df[df["overlap(m)"] > min_overlap].copy()

print(f"Remaining samples: {len(df)}")

# ============================================================
# INPUT FEATURES
#
# IMPORTANT:
# Inputs remain in PHYSICAL SPACE
#
# [ position_z , qw , qx , qy , qz ]
#
# DO NOT SCALE QUATERNIONS
# ============================================================

position_z = df["position_z"].values.astype(np.float32)

qw = df["Q_orientation.w()"].values.astype(np.float32)
qx = df["Q_orientation.x()"].values.astype(np.float32)
qy = df["Q_orientation.y()"].values.astype(np.float32)
qz = df["Q_orientation.z()"].values.astype(np.float32)

# ============================================================
# QUATERNION CANONICALIZATION
# ============================================================

quats = np.stack([qw, qx, qy, qz], axis=1)

mask = quats[:,0] < 0.0

quats[mask] *= -1.0

# ============================================================
# FINAL INPUT MATRIX
# ============================================================

X = np.column_stack([
    position_z,
    quats
]).astype(np.float32)

print("\nInput shape:", X.shape)

# ============================================================
# TARGETS
# ============================================================

overlap = df[["overlap(m)"]].values.astype(np.float32)

eps = 1e-15

overlap_log = np.log10(overlap + eps)

normal = df[[
    "normal_x",
    "normal_y",
    "normal_z"
]].values.astype(np.float32)

# ------------------------------------------------------------
# final target:
# [ log_overlap , nx , ny , nz ]
# ------------------------------------------------------------

y = np.hstack([
    overlap_log,
    normal
]).astype(np.float32)

print("Target shape:", y.shape)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# ============================================================
# NORMALIZE OUTPUTS ONLY
# ============================================================

y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(y_train_raw)
y_test  = y_scaler.transform(y_test_raw)

# ============================================================
# TENSORS
# ============================================================

X_train = torch.tensor(X_train_raw, dtype=torch.float32)
X_test  = torch.tensor(X_test_raw,  dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

# ============================================================
# SINE ACTIVATION
# ============================================================

class Sine(nn.Module):

    def forward(self, x):

        return torch.sin(x)

# ============================================================
# GEOMETRY-INFORMED MODEL
# ============================================================

class GeometryAwareModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Fixed cube geometry
        # ----------------------------------------------------

        half_size = 0.5

        vertices = torch.tensor([
            [-half_size, -half_size, -half_size],
            [-half_size, -half_size,  half_size],
            [-half_size,  half_size, -half_size],
            [-half_size,  half_size,  half_size],
            [ half_size, -half_size, -half_size],
            [ half_size, -half_size,  half_size],
            [ half_size,  half_size, -half_size],
            [ half_size,  half_size,  half_size]
        ], dtype=torch.float32)

        self.register_buffer(
            "local_vertices",
            vertices
        )

        # ----------------------------------------------------
        # Descriptor normalization
        # ----------------------------------------------------

        self.descriptor_mean = nn.Parameter(
            torch.zeros(8),
            requires_grad=False
        )

        self.descriptor_std = nn.Parameter(
            torch.ones(8),
            requires_grad=False
        )

        # ----------------------------------------------------
        # Shared feature extractor
        # ----------------------------------------------------

        self.features = nn.Sequential(

            nn.Linear(12, 128),
            Sine(),

            nn.Linear(128, 64),
            Sine(),

            nn.Linear(64, 32),
            Sine()
        )

        # ----------------------------------------------------
        # Overlap head
        # ----------------------------------------------------

        self.overlap_head = nn.Sequential(

            nn.Linear(32, 16),
            Sine(),

            nn.Linear(16, 1)
        )

        # ----------------------------------------------------
        # Normal head
        # ----------------------------------------------------

        self.normal_head = nn.Sequential(

            nn.Linear(32, 16),
            Sine(),

            nn.Linear(16, 3)
        )

    # ========================================================
    # QUATERNION → ROTATION MATRIX
    # ========================================================

    def quat_to_rotmat(self, q):

        qw = q[:,0]
        qx = q[:,1]
        qy = q[:,2]
        qz = q[:,3]

        R = torch.zeros(
            (q.shape[0], 3, 3),
            device=q.device
        )

        R[:,0,0] = 1 - 2*(qy*qy + qz*qz)
        R[:,0,1] = 2*(qx*qy - qz*qw)
        R[:,0,2] = 2*(qx*qz + qy*qw)

        R[:,1,0] = 2*(qx*qy + qz*qw)
        R[:,1,1] = 1 - 2*(qx*qx + qz*qz)
        R[:,1,2] = 2*(qy*qz - qx*qw)

        R[:,2,0] = 2*(qx*qz - qy*qw)
        R[:,2,1] = 2*(qy*qz + qx*qw)
        R[:,2,2] = 1 - 2*(qx*qx + qy*qy)

        return R

    # ========================================================
    # BUILD SUPPORT DESCRIPTOR
    # ========================================================

    def build_descriptor(self, x):

        position_z = x[:,0:1]

        quat = x[:,1:5]

        quat = quat / (
            torch.norm(
                quat,
                dim=1,
                keepdim=True
            ) + 1e-12
        )

        R = self.quat_to_rotmat(quat)

        vertices = self.local_vertices.unsqueeze(0)

        rotated = torch.matmul(
            vertices,
            R.transpose(1,2)
        )

        zvals = rotated[:,:,2]

        zvals = zvals + position_z

        # ----------------------------------------------------
        # Permutation-invariant support descriptor
        # ----------------------------------------------------

        zvals, _ = torch.sort(zvals, dim=1)

        return zvals

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        zvals = self.build_descriptor(x)

        zvals = (
            zvals - self.descriptor_mean
        ) / self.descriptor_std

        quat = x[:,1:5]

        quat = quat / (
            torch.norm(
                quat,
                dim=1,
                keepdim=True
            ) + 1e-12
        )

        # ----------------------------------------------------
        # Hybrid descriptor
        # ----------------------------------------------------

        features = torch.cat(
            [zvals, quat],
            dim=1
        )

        latent = self.features(features)

        overlap = self.overlap_head(latent)

        normal = self.normal_head(latent)

        # ----------------------------------------------------
        # Normalize predicted normal
        # ----------------------------------------------------

        normal = normal / (
            torch.norm(
                normal,
                dim=1,
                keepdim=True
            ) + 1e-12
        )

        y = torch.cat(
            [overlap, normal],
            dim=1
        )

        return y

# ============================================================
# MODEL
# ============================================================

model = GeometryAwareModel()

# ============================================================
# DESCRIPTOR NORMALIZATION
# ============================================================

with torch.no_grad():

    descriptors = model.build_descriptor(X_train)

    descriptor_mean = descriptors.mean(dim=0)

    descriptor_std = descriptors.std(dim=0)

    model.descriptor_mean.copy_(descriptor_mean)

    model.descriptor_std.copy_(descriptor_std)

# ============================================================
# LOSS
# ============================================================

mse_loss = nn.MSELoss()

normal_weight = 0.1

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-7
)

# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=200
)

# ============================================================
# TRAINING
# ============================================================

epochs = EPOCHS

best_test_loss = np.inf

patience = PATIENCE

counter = 0

train_losses = []
test_losses  = []

for epoch in range(epochs):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_pred = model(X_train)

    overlap_loss = mse_loss(
        train_pred[:,0:1],
        y_train[:,0:1]
    )

    normal_loss = mse_loss(
        train_pred[:,1:4],
        y_train[:,1:4]
    )

    train_loss = (
        overlap_loss
        + normal_weight * normal_loss
    )

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        test_pred = model(X_test)

        overlap_loss_test = mse_loss(
            test_pred[:,0:1],
            y_test[:,0:1]
        )

        normal_loss_test = mse_loss(
            test_pred[:,1:4],
            y_test[:,1:4]
        )

        test_loss = (
            overlap_loss_test
            + normal_weight * normal_loss_test
        )

    scheduler.step(test_loss)

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if test_loss.item() < best_test_loss:

        best_test_loss = test_loss.item()

        torch.save(
            model.state_dict(),
            "best_overlap_model.pth"
        )

        counter = 0

    else:

        counter += 1

    if counter >= patience:

        print(f"\nEarly stopping at epoch {epoch}")

        break

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    if epoch % 100 == 0:

        lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:5d} | "
            f"Train = {train_loss.item():.6e} | "
            f"Test = {test_loss.item():.6e} | "
            f"LR = {lr:.2e}"
        )

# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load("best_overlap_model.pth")
)

# ============================================================
# FINAL EVALUATION
# ============================================================

model.eval()

with torch.no_grad():

    train_pred_all = model(X_train).numpy()

    test_pred_all = model(X_test).numpy()

# ============================================================
# INVERSE SCALE
# ============================================================

train_pred_all = y_scaler.inverse_transform(train_pred_all)

test_pred_all = y_scaler.inverse_transform(test_pred_all)

train_true_all = y_scaler.inverse_transform(
    y_train.numpy()
)

test_true_all = y_scaler.inverse_transform(
    y_test.numpy()
)

# ============================================================
# OVERLAP RECOVERY
# ============================================================

train_pred_log = train_pred_all[:,0:1]
test_pred_log  = test_pred_all[:,0:1]

train_true_log = train_true_all[:,0:1]
test_true_log  = test_true_all[:,0:1]

train_pred = 10.0**train_pred_log - eps
test_pred  = 10.0**test_pred_log  - eps

train_true = 10.0**train_true_log - eps
test_true  = 10.0**test_true_log  - eps

# ============================================================
# OVERLAP METRICS
# ============================================================

test_rmse = np.sqrt(
    np.mean((test_pred - test_true)**2)
)

print("\n================================================")
print("OVERLAP RESULTS")
print("================================================")

print(f"Test RMSE = {test_rmse:.6e}")

# ============================================================
# NORMAL DIRECTION EVALUATION
# ============================================================

train_pred_normals = train_pred_all[:,1:4]
test_pred_normals  = test_pred_all[:,1:4]

train_true_normals = train_true_all[:,1:4]
test_true_normals  = test_true_all[:,1:4]

# ------------------------------------------------------------
# Normalize vectors
# ------------------------------------------------------------

train_pred_normals /= (
    np.linalg.norm(
        train_pred_normals,
        axis=1,
        keepdims=True
    ) + 1e-12
)

test_pred_normals /= (
    np.linalg.norm(
        test_pred_normals,
        axis=1,
        keepdims=True
    ) + 1e-12
)

train_true_normals /= (
    np.linalg.norm(
        train_true_normals,
        axis=1,
        keepdims=True
    ) + 1e-12
)

test_true_normals /= (
    np.linalg.norm(
        test_true_normals,
        axis=1,
        keepdims=True
    ) + 1e-12
)

# ------------------------------------------------------------
# Cosine similarity
# ------------------------------------------------------------

train_cosine = np.sum(
    train_pred_normals * train_true_normals,
    axis=1
)

test_cosine = np.sum(
    test_pred_normals * test_true_normals,
    axis=1
)

train_cosine = np.clip(train_cosine, -1.0, 1.0)
test_cosine  = np.clip(test_cosine,  -1.0, 1.0)

# ------------------------------------------------------------
# Angular error
# ------------------------------------------------------------

train_angle_deg = np.degrees(
    np.arccos(train_cosine)
)

test_angle_deg = np.degrees(
    np.arccos(test_cosine)
)

print("\n================================================")
print("NORMAL RESULTS")
print("================================================")

print(
    f"Test Mean Angular Error = "
    f"{test_angle_deg.mean():.4f} deg"
)

print(
    f"Test Median Angular Error = "
    f"{np.median(test_angle_deg):.4f} deg"
)

# ============================================================
# LOSS CURVES
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Train")
plt.plot(test_losses, label="Test")

plt.yscale("log")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training History")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "loss_curve.png",
    dpi=200
)

plt.show()

# ============================================================
# NORMAL ERROR HISTOGRAM
# ============================================================

plt.figure(figsize=(8,5))

plt.hist(
    test_angle_deg,
    bins=100,
    alpha=0.7
)

plt.xlabel("Angular Error [deg]")
plt.ylabel("Count")

plt.title("Normal Direction Error")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "normal_error_histogram.png",
    dpi=200
)

plt.show()

# ============================================================
# NORMAL COSINE SIMILARITY
# ============================================================

plt.figure(figsize=(8,5))

plt.scatter(
    np.arange(len(test_cosine)),
    test_cosine,
    s=5,
    alpha=0.5
)

plt.axhline(
    1.0,
    color='k',
    linestyle='--'
)

plt.xlabel("Sample")
plt.ylabel("Cosine Similarity")

plt.title("Predicted vs True Normal Similarity")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "normal_cosine_similarity.png",
    dpi=200
)

plt.show()

# ============================================================
# PREDICTED VS TRUE OVERLAP
# ============================================================

plt.figure(figsize=(7,7))

plt.scatter(
    train_true,
    train_pred,
    s=8,
    alpha=0.4,
    label="Train"
)

plt.scatter(
    test_true,
    test_pred,
    s=8,
    alpha=0.4,
    label="Test"
)

vmin = min(
    train_true.min(),
    test_true.min(),
    train_pred.min(),
    test_pred.min()
)

vmax = max(
    train_true.max(),
    test_true.max(),
    train_pred.max(),
    test_pred.max()
)

plt.plot(
    [vmin, vmax],
    [vmin, vmax],
    'k--'
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("True overlap")
plt.ylabel("Predicted overlap")

plt.title("Predicted vs True")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "predicted_vs_true_overlap.png",
    dpi=200
)

plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "overlap_model.pth"
)

example = torch.randn(1,5)

traced_model = torch.jit.trace(
    model,
    example
)

traced_model.save(
    "overlap_model.pt"
)

# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    y_scaler,
    "y_scaler.save"
)

print("\nModels saved.")

with open("scalers.dat", "w") as f:

    f.write("y_mean\n")
    f.write(" ".join(map(str, y_scaler.mean_)) + "\n")

    f.write("y_scale\n")
    f.write(" ".join(map(str, y_scaler.scale_)) + "\n")

    f.write("eps\n")
    f.write(f"{eps}\n")

print("scalers.dat saved.")



print("\n================================================")
print("COPY INTO C++")
print("================================================")

print(
    "std::vector<double> y_mean = {"
    + ", ".join(map(str, y_scaler.mean_))
    + "};"
)

print(
    "std::vector<double> y_scale = {"
    + ", ".join(map(str, y_scaler.scale_))
    + "};"
)
