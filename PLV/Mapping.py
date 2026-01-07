

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial import cKDTree

'''
Yeo7 atlas mapping function.
'''

YEO7_LABELS = {
    0: "Unlabeled",
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal Attention",
    4: "Ventral Attention / Salience (CON)",
    5: "Limbic",
    6: "Frontoparietal (FPN)",
    7: "Default Mode (DMN)",
}

YEO7_COLORS = {
    0: "gray",
    1: "green",
    2: "blue",
    3: "gold",
    4: "purple",   # CON
    5: "brown",
    6: "orange",   # FPN
    7: "red",      # DMN
}

def laterality_from_mni_x(x_mm: float, tol: float = 2.0) -> str:
    if abs(x_mm) <= tol:
        return "Midline"
    return "Left" if x_mm < 0 else "Right"

def mode_int(x: np.ndarray) -> int:
    """Most frequent integer in x (ties -> first by np.argmax)."""
    if x.size == 0:
        return 0
    vals, counts = np.unique(x.astype(int), return_counts=True)
    return int(vals[np.argmax(counts)])

def map_electrodes_to_yeo7(
    chan_names: list[str],
    mni_xyz_mm: np.ndarray,          # (N,3) electrode coords in MNI mm
    atlas_nii_path: str,             # Yeo atlas nifti
    range_mm: float = 10.0,          # MATLAB: range = 10 (mm)
) -> pd.DataFrame:

    # --- inputs ---
    mni_xyz_mm = np.asarray(mni_xyz_mm, dtype=float)
    if mni_xyz_mm.ndim != 2 or mni_xyz_mm.shape[1] != 3:
        raise ValueError("mni_xyz_mm must be (N,3)")
    if len(chan_names) != mni_xyz_mm.shape[0]:
        raise ValueError("chan_names and mni_xyz_mm length mismatch")

    # --- load atlas ---
    img = nib.load(atlas_nii_path)
    data = np.asarray(img.get_fdata())
    affine = np.asarray(img.affine)

    # atlas can be (X,Y,Z,1) -> squeeze to (X,Y,Z)
    if data.ndim == 4:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D atlas, got shape {data.shape}")

    data = data.astype(int)
    if affine.shape != (4, 4):
        affine = affine[:4, :4]

    # --- build labeled-voxel point cloud (voxel ijk -> MNI xyz mm) ---
    # ijk: indices of voxels where label>0 ; labels: corresponding label at each ijk
    ijk = np.column_stack(np.where(data > 0))          # (n_vox, 3)
    labels = data[data > 0].astype(int)                # (n_vox,)

    xyz = nib.affines.apply_affine(affine, ijk)        # (n_vox, 3) in MNI mm

    # KD-tree speeds up radius queries (equivalent to MATLAB dist computation)
    tree = cKDTree(xyz)

    out_label = np.zeros(len(chan_names), dtype=int)
    out_min_dist = np.zeros(len(chan_names), dtype=float)

    for i, elec_xyz in enumerate(mni_xyz_mm):
        # MATLAB: min(dist)
        min_dist, nn_idx = tree.query(elec_xyz, k=1)
        out_min_dist[i] = float(min_dist)

        # MATLAB: labels_within = labels(dist > min_dist & dist < min_dist + range)
        r = float(min_dist + range_mm)
        cand = tree.query_ball_point(elec_xyz, r=r)    # includes dist <= r

        if len(cand) == 0:
            out_label[i] = 0
            continue

        cand = np.asarray(cand, dtype=int)

        # inequality filtering
        d = np.linalg.norm(xyz[cand] - elec_xyz[None, :], axis=1)
        keep = (d > min_dist) & (d < (min_dist + range_mm))

        if not np.any(keep):
            out_label[i] = 0
            continue

        out_label[i] = mode_int(labels[cand[keep]])
    
    laterality = [laterality_from_mni_x(x) for x in mni_xyz_mm[:, 0]]

    df = pd.DataFrame({
        "channel": list(chan_names),
        "x": mni_xyz_mm[:, 0],
        "y": mni_xyz_mm[:, 1],
        "z": mni_xyz_mm[:, 2],
        "laterality": laterality,
        "min_dist_mm": out_min_dist,
        "yeo7_label": out_label,
        "yeo7_network": [YEO7_LABELS.get(int(l), "Unknown") for l in out_label],
        "yeo7_color": [YEO7_COLORS.get(int(l), "gray") for l in out_label],
    })

    return df



def _electrode_xyz_for_display(
    df: pd.DataFrame,
    mirror_all_to_left: bool = False,
    only_left_hemisphere: bool = False,
) -> np.ndarray:
    """
    Returns electrode xyz for plotting.
    - mirror_all_to_left: x -> -abs(x) for all electrodes (puts everything on left)
    - only_left_hemisphere: keep only electrodes with x<0 (after optional mirroring)
    """
    xyz = df[["x", "y", "z"]].to_numpy(dtype=float).copy()

    if mirror_all_to_left:
        xyz[:, 0] = -np.abs(xyz[:, 0])

    if only_left_hemisphere:
        keep = xyz[:, 0] < 0
        xyz = xyz[keep]

    return xyz


import numpy as np
import pandas as pd
import nibabel as nib

def visualize_on_atlas_surface_plotly(
    atlas_nii_path: str,
    df: pd.DataFrame,
    mode: str = "network",   # "network" or "laterality"
    mirror_all_to_left: bool = False,
    only_left_hemisphere: bool = False,
    surface_from: str = "atlas_nonzero",  # "atlas_nonzero" or "atlas_label_1plus"
    point_size: int = 6,
    title: str | None = None,
):
    """
    Interactive 3D Plotly: atlas-derived surface + electrode dots.
    Adds titles + legends (per network or per laterality).

    Notes:
      - Uses df columns: x,y,z, channel, yeo7_network, yeo7_label, laterality, yeo7_color (optional)
      - If yeo7_color missing, falls back to mapping from YEO7_COLORS using yeo7_label.
    """
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes

    # ---------- make a display copy (mirror/filter) ----------
    df_disp = df.copy()

    if mirror_all_to_left:
        df_disp["x"] = -np.abs(df_disp["x"].to_numpy(float))

    if only_left_hemisphere:
        df_disp = df_disp[df_disp["x"] < 0].copy()

    # ---------- load atlas + build surface mesh ----------
    img = nib.load(atlas_nii_path)
    vol = np.asarray(img.get_fdata())
    aff = np.asarray(img.affine)

    if vol.ndim == 4:
        vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D atlas, got shape {vol.shape}")

    vol = vol.astype(float)

    if surface_from == "atlas_label_1plus":
        mask = (vol >= 1).astype(np.uint8)
    else:
        mask = (vol > 0).astype(np.uint8)

    verts_ijk, faces, _, _ = marching_cubes(mask, level=0.5)
    verts_xyz = nib.affines.apply_affine(aff, verts_ijk)

    mesh = go.Mesh3d(
        x=verts_xyz[:, 0], y=verts_xyz[:, 1], z=verts_xyz[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        opacity=0.15,
        name="Brain surface",
        color="lightgray",
        flatshading=True,
        showlegend=True,
    )

    # ---------- hover text ----------
    hover = (
        df_disp["channel"].astype(str)
        + "<br>" + df_disp.get("yeo7_network", "Unknown").astype(str)
        + "<br>" + df_disp.get("laterality", "Unknown").astype(str)
    )

    # ---------- choose traces + legend grouping ----------
    traces = [mesh]

    if mode == "laterality":
        # Left/Right/Midline legend
        lat_to_color = {"Left": "blue", "Right": "red", "Midline": "gray"}
        order = ["Left", "Right", "Midline"]

        for lat in order:
            sub = df_disp[df_disp["laterality"] == lat]
            if sub.empty:
                continue
            traces.append(
                go.Scatter3d(
                    x=sub["x"], y=sub["y"], z=sub["z"],
                    mode="markers",
                    name=f"{lat} hemisphere",
                    marker=dict(size=point_size, color=lat_to_color[lat], opacity=0.95),
                    text=hover.loc[sub.index],
                    hoverinfo="text",
                    showlegend=True,
                )
            )

        default_title = "Electrodes colored by laterality (Right=red, Left=blue)"

    else:
        # Network legend (one trace per network)
        # If yeo7_color not present, compute it from YEO7_COLORS using yeo7_label
        if "yeo7_color" not in df_disp.columns:
            # expects YEO7_COLORS dict in your workspace
            df_disp["yeo7_color"] = df_disp["yeo7_label"].astype(int).map(YEO7_COLORS).fillna("gray")

        # keep a stable legend order by label number if possible
        # build mapping network->(label,color)
        tmp = df_disp[["yeo7_label", "yeo7_network", "yeo7_color"]].drop_duplicates()
        tmp = tmp.sort_values("yeo7_label")

        for _, row in tmp.iterrows():
            net = str(row["yeo7_network"])
            col = str(row["yeo7_color"])
            sub = df_disp[df_disp["yeo7_network"] == net]
            if sub.empty:
                continue

            traces.append(
                go.Scatter3d(
                    x=sub["x"], y=sub["y"], z=sub["z"],
                    mode="markers",
                    name=net,
                    marker=dict(size=point_size, color=col, opacity=0.95),
                    text=hover.loc[sub.index],
                    hoverinfo="text",
                    showlegend=True,
                )
            )

        default_title = "Electrodes colored by Yeo7 networks"

    # ---------- figure ----------
    fig = go.Figure(data=traces)

    fig.update_layout(
        title=(title if title is not None else default_title),
        scene=dict(
            xaxis_title="MNI X (mm)",
            yaxis_title="MNI Y (mm)",
            zaxis_title="MNI Z (mm)",
            aspectmode="data",
        ),
        legend=dict(
            title="Legend",
            itemsizing="constant",
            borderwidth=0,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    fig.show()


def visualize_simple_matplotlib(
    df: pd.DataFrame,
    mode: str = "network",  # "network" or "laterality"
    mirror_all_to_left: bool = False,
    only_left_hemisphere: bool = False,
    point_size: int = 18,
):
    """
    Fallback: just a 3D scatter (no surface), Matplotlib only.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    tmp = df.copy()
    if mirror_all_to_left:
        tmp["x"] = -np.abs(tmp["x"])
    if only_left_hemisphere:
        tmp = tmp[tmp["x"] < 0].copy()

    if mode == "laterality":
        c = tmp["laterality"].map({"Left": "blue", "Right": "red", "Midline": "gray"}).fillna("gray").tolist()
        title = "Electrodes (laterality): Left=blue, Right=red"
    else:
        c = tmp["yeo7_color"].tolist()
        title = "Electrodes (Yeo7 networks)"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tmp["x"], tmp["y"], tmp["z"], s=point_size, c=c, depthshade=True)
    ax.set_title(title)
    ax.set_xlabel("MNI X (mm)")
    ax.set_ylabel("MNI Y (mm)")
    ax.set_zlabel("MNI Z (mm)")
    plt.show()