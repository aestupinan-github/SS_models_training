import os
import random
import numpy as np
import pandas as pd
import pyvista as pv

from scipy.spatial.transform import Rotation as R

# ============================================================
# PARAMETERS
# ============================================================

# cube side [m]
L = 1#5e-3

MIN_OVERLAP = 1e-2
MAX_OVERLAP = 0.5#1e-2

N_SAMPLES = 10000

OUTPUT_CSV = "training_data.csv"

VIS_FOLDER = "paraview_samples_vertex"

N_VISUALIZATION = 10

# ============================================================
# GEOMETRY LIMITS
# ============================================================

# minimum center height for possible contact
# cube face parallel to plane

Z_MIN = L / 2.0

# maximum center height for possible contact
# cube touching by a corner

Z_MAX = np.sqrt(3.0) * L / 2.0

print(f"Z_MIN = {Z_MIN:.6e}")
print(f"Z_MAX = {Z_MAX:.6e}")

# ============================================================
# FUNCTIONS
# ============================================================

def compute_projected_half_height(rotation_matrix, L):

    return (L / 2.0) * (
        abs(rotation_matrix[2, 0]) +
        abs(rotation_matrix[2, 1]) +
        abs(rotation_matrix[2, 2])
    )

# ============================================================
# GENERATE DATASET
# ============================================================

dataset = []

while len(dataset) < N_SAMPLES:

    # --------------------------------------------------------
    # RANDOM POSITION FIRST
    # --------------------------------------------------------

    z_center = np.random.uniform(Z_MIN, Z_MAX)

    # --------------------------------------------------------
    # RANDOM ROTATION
    # --------------------------------------------------------

    rotation = R.random()

    rotation_matrix = rotation.as_matrix()

    qx, qy, qz, qw = rotation.as_quat()

    # --------------------------------------------------------
    # COMPUTE CONTACT GEOMETRY
    # --------------------------------------------------------

    projected_half_height = compute_projected_half_height(
        rotation_matrix,
        L
    )

    clearance = projected_half_height - z_center

    overlap = max(0.0, clearance)

    # --------------------------------------------------------
    # FILTER VALID OVERLAPS
    # --------------------------------------------------------

    if MIN_OVERLAP <= overlap <= MAX_OVERLAP:

        # interaction normal in cube local frame
        normal_local = rotation_matrix.T @ np.array([0.0, 0.0, 1.0])

        nx = normal_local[0]
        ny = normal_local[1]
        nz = normal_local[2]

        dataset.append([
            overlap,
            clearance,
            z_center,
            qw,
            qx,
            qy,
            qz,
            nx,
            ny,
            nz
        ])

# ============================================================
# SAVE CSV
# ============================================================

columns = [
    "overlap(m)",
    "clearance(m)",
    "position_z",
    "Q_orientation.w()",
    "Q_orientation.x()",
    "Q_orientation.y()",
    "Q_orientation.z()",
    "normal_x",
    "normal_y",
    "normal_z"
]

df = pd.DataFrame(dataset, columns=columns)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    float_format="%.16e"
)

print(f"\nSaved dataset: {OUTPUT_CSV}")

print(df.describe())

# ============================================================
# PARAVIEW VISUALIZATION EXPORT
# ============================================================

os.makedirs(VIS_FOLDER, exist_ok=True)

# save plane

plane = pv.Plane(
    center=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    i_size=0.03,
    j_size=0.03
)

plane.save(os.path.join(VIS_FOLDER, "plane.vtp"))

# select random samples

indices = random.sample(range(len(df)), N_VISUALIZATION)

for i, idx in enumerate(indices):

    row = df.iloc[idx]

    z_center = row["position_z"]

    qw = row["Q_orientation.w()"]
    qx = row["Q_orientation.x()"]
    qy = row["Q_orientation.y()"]
    qz = row["Q_orientation.z()"]

    # scipy expects [x,y,z,w]
    rotation = R.from_quat([qx, qy, qz, qw])

    rotation_matrix = rotation.as_matrix()

    # create cube

    cube = pv.Cube(
        center=(0.0, 0.0, 0.0),
        x_length=L,
        y_length=L,
        z_length=L
    )

    # apply rotation

    transform = np.eye(4)

    transform[:3, :3] = rotation_matrix

    cube.transform(transform, inplace=True)

    # translate

    cube.translate(
        (0.0, 0.0, z_center),
        inplace=True
    )

    # save

    filename = os.path.join(
        VIS_FOLDER,
        f"cube_sample_{i:03d}.vtp"
    )

    cube.save(filename)

    print(
        f"Saved visualization sample: {filename}"
    )

print("\nVisualization files ready for ParaView.")
