#%%

"""Offline helper: export a THOR ViT encoder to ONNX.

Run cell-by-cell. Set ``MODEL`` in the config cell and execute.

THOR (https://github.com/FM4CS/THOR) is a compute-adaptive foundation model that
unifies Sentinel-1, -2 and -3 in a single FlexiViT backbone with per-band-group
patch embeddings. This example uses all three: Sentinel-2 L2A (BLUE, GREEN, RED,
NIR, 10 m), Sentinel-1 GRD (VV, VH, 10 m), Sentinel-3 OLCI (21 reflectance bands,
240 m) and Sentinel-3 SLSTR (6 reflectance + 3 brightness-temperature bands,
480/960 m) — with ``merge_method="mean"`` so every group's token grid is
interpolated to the same spatial resolution and averaged into a single
``(B, D, H, W)`` feature map per encoder block. The default ``GROUND_COVER``/
``PATCH_SIZES`` below use a lighter ~9.6 km tile (960x960 px S1/S2 side) chosen
for manageable compute cost; THOR's own multi-sensor reference example
(``10_multimodal_thor_inference.ipynb`` in ``thor_terratorch_ext``) instead uses
a much larger 26.88 km/2688x2688 px tile, since a coarse sensor like OLCI
(240 m GSD) needs a large ground cover just to span a handful of patches — that
reference config is kept below as a commented-out alternative. See the README's
Caveats for the compute-cost trade-off and the S3 token-grid impact of the
smaller default.

The exported graph:
    - takes a single input ``x (B, C, H, W)`` with channels in the exact order
      of ``backbone.bands`` (36 channels for this config: 2 S1 + 4 S2 + 21
      OLCI + 6 SLSTR-reflectance + 3 SLSTR-BT; printed below, and saved to
      ``bands_order.json`` next to the ``.onnx`` file),
    - applies THOR's published pretraining per-band mean/std internally (baked
      as Constants in the ONNX graph, taken directly from
      ``thor_terratorch_ext``'s ``THOR_NORMALIZATION_PARAMS``, so the UDF
      doesn't need to know the values),
    - returns the last encoder block's merged feature map ``(B, D, H, W)``.

What stays dynamic after export, and what doesn't
--------------------------------------------------
THOR's headline feature is compute-adaptive resolution: you pick a
``ground_cover`` (metres) and ``patch_sizes`` (pixels) and the FlexiViT
backbone adapts its patch embedding/position encoding accordingly. Internally,
``THOREncoderWrapper._preprocess_input`` always bilinear-resizes whatever
tensor you pass it to a *fixed* pixel size (``ground_cover / GSD``, computed
at build time) before patch embedding — it does **not** derive that size from
the input tensor's actual shape. Two consequences for the ONNX export:

1. Input ``x``'s spatial dims (``H``, ``W``) *are* declared dynamic below,
   and that's safe/useful: whatever tile shape ``apply_neighborhood`` hands
   the UDF (including ragged edge tiles) gets resized internally to the
   model's fixed working resolution, so the UDF isn't forced to pad every
   tile to an exact pixel count.
2. The actual resolution/compute trade-off (``GROUND_COVER``,
   ``PATCH_SIZE``) is baked into the traced graph's weights at *export* time
   and cannot be changed at inference. To get a different
   resolution/compute profile, re-run this script with different
   ``GROUND_COVER``/``PATCH_SIZE`` values and ship the resulting archive
   instead — one exported ``.onnx`` per profile. 

Requires (offline only): ``thor`` + ``thor_terratorch_ext`` + ``terratorch`` +
``torch`` + ``onnx`` installed, per https://github.com/FM4CS/THOR and
https://github.com/FM4CS/thor_terratorch_ext.

Then upload the ``.onnx`` (+ its ``.onnx.data`` sidecar if present, and
``bands_order.json``) to a location the openEO backend can reach, zip them
together, and pass the URL via the ``udf-dependency-archives`` job option.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn


class THOREncoderONNX(nn.Module):
    """Standardize (baked THOR pretraining stats) + backbone(mean-merge) + last block."""

    def __init__(self, backbone: nn.Module, means, stds):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(stds, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out   # (B, D, H, W)


# %% Config — edit and re-run
MODEL = "thor_v1_base"   # thor_v1_tiny / thor_v1_small / thor_v1_base / thor_v1_large

# GROUND_COVER must divide evenly by every modality's native GSD (10 m for
# S1/S2, 240 m OLCI, 480/960 m SLSTR) so each modality's working pixel size
# (ground_cover / gsd) and its patch_sizes entry both stay whole numbers —
# otherwise you get a fractional token grid for that modality.
#
# Option A (default): a lighter ~9.6 km tile (960x960 px S1/S2 side) —
# chosen over THOR's much larger reference config for manageable compute
# cost. NOT from THOR's reference example — untested against
# thor_terratorch_ext beyond the pixel-math constraint above. S3 sensors
# only resolve to a 5x5 token grid each (vs. 7x7 in the reference config).
GROUND_COVER = 9600     # metres (9.6 km); 960x960 px S1/S2 tile
PATCH_SIZES = {
    "S1GRD": 16,          # 960 px / 32 = 30 tokens/side
    "S2L2A": 16,          # 30 tokens/side
    "S3OLCI": 4,          # 40 px / 8 = 5 tokens/side
    "S3SLSTR_REFL": 2,    # 20 px / 4 = 5 tokens/side
    "S3SLSTR_BT": 2,      # 10 px / 2 = 5 tokens/side
}

# Option B: THOR's own multi-sensor reference config, taken verbatim from
# thor_terratorch_ext's example notebook (10_multimodal_thor_inference.ipynb)
# — not invented for this example. Much heavier: ~1 GB per UDF invocation
# (see README Caveats). Uncomment to use instead of the lighter default above.
# GROUND_COVER = 26880   # metres (26.88 km); at 10 m GSD this is a 2688x2688 px S1/S2 tile
# PATCH_SIZES = {
#     "S1GRD": 32,
#     "S2L2A": 32,
#     "S3OLCI": 16,
#     "S3SLSTR_REFL": 8,
#     "S3SLSTR_BT": 4,
# }

# This only matters if a PATCH_SIZES entry above is a *list* of candidate patch
# sizes (letting THOR pick one adaptively) rather than the single fixed ints
# used above/below — with a single int per group there's only one choice, so
# this is currently a no-op. "min"/"max" pick the smallest/largest patch size
# per group independently; "equal-min"/"equal-max" instead require all groups
# to resolve to the same token-grid size (erroring if that's not possible).
SELECT_PATCH_STRATEGY = "min"
OUT_PATH = Path(f"{MODEL}_encoder.onnx")
OPSET = 17

# %% Build backbone (downloads pretrained weights from the FM4CS Hugging Face hub)
import thor_terratorch_ext  # noqa: F401  (registers THOR backbones with terratorch)
from terratorch.registry import BACKBONE_REGISTRY
from thor_terratorch_ext.datasets.utils import S2L2ABands, S3OLCIBands, S3SLSTRBands, SARThorBands
from thor_terratorch_ext.models.backbones.thor_vit import THOR_NORMALIZATION_PARAMS

SLSTR_REFL_BANDS = [
    S3SLSTRBands.S1_REFLECTANCE_AN, S3SLSTRBands.S2_REFLECTANCE_AN,
    S3SLSTRBands.S3_REFLECTANCE_AN, S3SLSTRBands.S4_REFLECTANCE_AN,
    S3SLSTRBands.S5_REFLECTANCE_AN, S3SLSTRBands.S6_REFLECTANCE_AN,
]
SLSTR_BT_BANDS = [S3SLSTRBands.S7_BT_IN, S3SLSTRBands.S8_BT_IN, S3SLSTRBands.S9_BT_IN]

MODALITIES = {
    "S1GRD": [SARThorBands.IW_VV, SARThorBands.IW_VH],
    "S2L2A": [S2L2ABands.BLUE, S2L2ABands.GREEN, S2L2ABands.RED, S2L2ABands.NIR_BROAD],
    "S3OLCI": list(S3OLCIBands),   # all 21 Oa01-Oa21 reflectance bands
    "S3SLSTR_REFL": SLSTR_REFL_BANDS,
    "S3SLSTR_BT": SLSTR_BT_BANDS,
}

print(f"Building {MODEL} from terratorch...")
backbone = BACKBONE_REGISTRY.build(
    MODEL,
    pretrained=True,
    modalities=MODALITIES,
    ground_cover=GROUND_COVER,
    patch_sizes=PATCH_SIZES,
    select_patch_strategy=SELECT_PATCH_STRATEGY,
    merge_method="mean",
).eval()

# ``backbone.bands`` is the resolved, internal-name channel order the model
# actually expects (e.g. ["S2:Blue", "S2:Green", "S2:Red", "S2:NIR", "S1:IW-VV",
# "S1:IW-VH"]). Look up pretraining mean/std per band directly from the
# package rather than hardcoding, so this stays correct if the internal
# ordering ever changes.
band_order = list(backbone.bands)
means = [THOR_NORMALIZATION_PARAMS[b]["mean"] for b in band_order]
stds = [THOR_NORMALIZATION_PARAMS[b]["std"] for b in band_order]
embed_dim = backbone.out_channels[0]

print("Resolved band order:", band_order, f"({len(band_order)} channels)")
print("Embedding dim:", embed_dim)
print("Lowest native GSD (m):", backbone.lowest_gsd)
print("Input size (px, at lowest GSD):", GROUND_COVER // backbone.lowest_gsd)

wrapped = THOREncoderONNX(backbone, means, stds).eval()

# %% Export to ONNX
n_channels = len(band_order)
input_px = GROUND_COVER // backbone.lowest_gsd
dummy = torch.zeros(1, n_channels, input_px, input_px)

print(f"Exporting to {OUT_PATH} (opset {OPSET})...")
torch.onnx.export(
    wrapped,
    (dummy,),
    OUT_PATH.as_posix(),
    input_names=["x"],
    output_names=["features"],
    # Batch and input H/W are dynamic: the model resizes any tile internally
    # to its fixed working resolution before patch embedding (see module
    # docstring), so the UDF can feed variable/ragged tile shapes. Output
    # H/W are NOT dynamic: the token grid size is fixed by GROUND_COVER /
    # PATCH_SIZE at export time, regardless of input tile shape.
    dynamic_axes={
        "x": {0: "batch", 2: "height", 3: "width"},
        "features": {0: "batch"},
    },
    opset_version=OPSET,
    do_constant_folding=True,
)
print(f"Done. Wrote {OUT_PATH.stat().st_size / 1e6:.1f} MB.")

# %% Save the resolved band order + config next to the ONNX file (the UDF needs this)
sidecar = OUT_PATH.with_name("bands_order.json")
sidecar.write_text(json.dumps({
    "band_order": band_order,
    "ground_cover": GROUND_COVER,
    "patch_sizes": PATCH_SIZES,
    "embed_dim": embed_dim,
}, indent=2))
print(f"Wrote {sidecar}")

# %%
