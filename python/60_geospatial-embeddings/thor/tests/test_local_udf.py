#%%
"""Local UDF test: run the THOR ONNX UDF on results/thor_input.nc.

Loads the merged S2 + S1 + S3 OLCI/SLSTR NetCDF produced by
``thor-embedding.ipynb`` (the whole AOI, already sized to match the exported
model's ground cover — no tiling needed), feeds it straight into
``udf_thor_embedding.apply_datacube``, and saves a PCA + K-means preview to
``tests/test_outputs/embedding_output.png``.

Requirements (installed in the local Python env, not via the openEO deps
archive)::

    pip install onnxruntime xarray netCDF4 matplotlib numpy scikit-learn openeo

This test expects the exported ONNX model + bands_order.json sidecar in
``thor_v1_base_onnx/`` (see ``export_thor_to_onnx.py``) and the input cube in
``results/thor_input.nc`` (produced by running the "Load and preprocess
Sentinel-1/2/3 data" cell in ``thor-embedding.ipynb``).

Usage::

    python tests/test_local_udf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _resolve_repo_root() -> Path:
    """Return the thor/ folder (the one holding the UDF + results/thor_input.nc)."""
    if "__file__" in globals():
        here = Path(__file__).resolve().parent
    else:
        here = Path.cwd().resolve()

    for candidate in (here, *here.parents):
        if (
            (candidate / "udf_thor_embedding.py").is_file()
            and (candidate / "results" / "thor_input.nc").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the thor/ folder (expected `udf_thor_embedding.py` + "
        f"`results/thor_input.nc` side by side). Searched from {here} upward."
    )


_REPO_ROOT = _resolve_repo_root()
sys.path.insert(0, str(_REPO_ROOT))

from udf_thor_embedding import apply_datacube
from openeo.udf import XarrayDataCube

# ---------------------------------------------------------------------------
# Config — tweak here
# ---------------------------------------------------------------------------
INPUT_NC = _REPO_ROOT / "results" / "thor_input.nc"
OUT_DIR = _REPO_ROOT / "tests" / "test_outputs"
LOCAL_ONNX_DIR = _REPO_ROOT / "thor_v1_base_gc9600_ps16-16-4-2-2"
LOCAL_ONNX_FILENAME = "thor_v1_base_encoder.onnx"
LOCAL_BANDS_JSON = "bands_order.json"

# Must match the AOI/acquisition window used to build results/thor_input.nc
# in thor-embedding.ipynb (needed for the single-point SZA approximation the
# UDF applies to the OLCI/SLSTR reflectance bands).
CENTER_LAT, CENTER_LON = 37.72, 15.00
ACQUISITION_DATETIME = "2018-07-15T10:30:00"

# UDF context: same keys the UDF reads on the backend (where the folder comes
# from a `udf-dependency-archives` alias instead of a local path).
CONTEXT = {
    "onnx_dir": str(LOCAL_ONNX_DIR),
    "onnx_filename": LOCAL_ONNX_FILENAME,
    "bands_json": LOCAL_BANDS_JSON,
    "center_lat": CENTER_LAT,
    "center_lon": CENTER_LON,
    "acquisition_datetime": ACQUISITION_DATETIME,
}


# ---------------------------------------------------------------------------
# Load thor_input.nc → (bands, y, x) DataArray in raw units
# ---------------------------------------------------------------------------

def load_input_cube(path: Path) -> xr.DataArray:
    """Return a (bands, y, x) DataArray from the merged S2 + S1 + S3 NetCDF."""
    ds = xr.open_dataset(path)
    print(f"Variables : {list(ds.data_vars)}")
    print(f"Dims      : {dict(ds.sizes)}")
    print(f"Coords    : {list(ds.coords)}")

    _NON_BAND_VARS = {"crs"}

    if "bands" in ds.dims:
        var_name = next(v for v in ds.data_vars if v not in _NON_BAND_VARS)
        da = ds[var_name]
    else:
        band_names = [
            v for v in ds.data_vars
            if v not in _NON_BAND_VARS
            and np.issubdtype(ds[v].dtype, np.number)
            and {"y", "x"}.issubset(set(ds[v].dims))
        ]
        if not band_names:
            raise ValueError(f"No band-like variables found in {path}")
        da = xr.concat([ds[v] for v in band_names], dim="bands")
        da = da.assign_coords(bands=band_names)

    t_dim = next((d for d in da.dims if d in ("t", "time")), None)
    if t_dim is not None:
        da = da.squeeze(t_dim) if da.sizes[t_dim] == 1 else da.median(dim=t_dim)

    da = da.astype(np.float32)

    print(f"Cube shape: {da.shape}, dims={list(da.dims)}, dtype={da.dtype}")
    if "bands" in da.dims:
        for i, name in enumerate(da.coords["bands"].values):
            v = da.isel(bands=i).values
            valid = v[~np.isnan(v)]
            if valid.size:
                print(
                    f"  band {name}: min={valid.min():.3f}, "
                    f"max={valid.max():.3f}, mean={valid.mean():.3f}"
                )
    return da



print("=" * 60)
print("LOCAL THOR UDF TEST")
print("=" * 60)
print(f"Loading {INPUT_NC}")

cube = load_input_cube(INPUT_NC)
print(f"\nCube: shape={cube.shape}, dims={list(cube.dims)}")

print("\nRunning UDF...")
result = apply_datacube(XarrayDataCube(cube), CONTEXT)
result_da = result.get_array() if hasattr(result, "get_array") else result
emb = result_da.values                          # (embed_dim, y, x)
print(
    f"Embedding: shape={emb.shape}, dtype={emb.dtype}, "
    f"min={emb.min():.3f}, max={emb.max():.3f}, mean={emb.mean():.3f}"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# S2 RGB preview from B04/B03/B02 (or first 3 bands as fallback).
band_names = list(cube.coords["bands"].values) if "bands" in cube.coords else []
rgb_bands = ["B04", "B03", "B02"]
if all(b in band_names for b in rgb_bands):
    s2_rgb = np.stack(
        [cube.sel(bands=b).values for b in rgb_bands], axis=-1
    ).astype(np.float32)
else:
    s2_rgb = cube.values[:3].transpose(1, 2, 0).astype(np.float32)

s2_rgb = np.nan_to_num(s2_rgb, nan=0.0)
s2_display = np.zeros_like(s2_rgb)
for i in range(3):
    band = s2_rgb[:, :, i]
    lo, hi = np.percentile(band, (2, 98))
    if hi > lo:
        s2_display[:, :, i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    else:
        s2_display[:, :, i] = band

# K-means on the D-dim tokens -> label per pixel (embedding is already
# upsampled to the input pixel grid by the UDF).
from sklearn.cluster import KMeans

n_clusters = 8
n_bands, H, W = emb.shape
tokens = emb.reshape(n_bands, -1).T             # (H*W, D)
labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(tokens)
label_map = labels.reshape(H, W)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"THOR embedding — {emb.shape[0]}-d tokens on a {H}x{W} grid", fontsize=13)

axes[0].imshow(s2_display)
axes[0].set_title("Input S2 RGB (2-98% stretch)")
axes[0].axis("off")

im = axes[1].imshow(label_map, cmap="tab20", interpolation="nearest",
                    vmin=0, vmax=max(n_clusters - 1, 1))
axes[1].set_title(f"K-means labels (k={n_clusters})")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(s2_display)
axes[2].imshow(label_map, cmap="tab20", alpha=0.55, interpolation="nearest",
                vmin=0, vmax=max(n_clusters - 1, 1))
axes[2].set_title(f"K-means on tokens (k={n_clusters}), overlay")
axes[2].axis("off")

plt.tight_layout()
out_png = OUT_DIR / "embedding_output.png"
plt.savefig(out_png, dpi=150)
print(f"\nSaved {out_png}")
plt.show()





# %%

result_da

# %%


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.suptitle(f"THOR embedding band 0 on a {H}x{W} grid", fontsize=13)

im = ax.imshow(result_da.isel(bands=2), cmap="viridis")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


plt.show()
