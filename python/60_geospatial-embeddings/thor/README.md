# THOR patch embeddings from openEO-loaded Sentinel-1/2/3

Computes [THOR](https://github.com/FM4CS/THOR) (Norwegian Computing Center + UiT + ESA
Φ-lab's compute-adaptive foundation model for Earth Observation) patch embeddings for a
custom AOI and time period directly from openEO. THOR unifies Sentinel-1 (SAR),
Sentinel-2 (MSI) and Sentinel-3 (OLCI/SLSTR) in a single FlexiViT backbone with
per-band-group patch embeddings and 2D ALiBi position encoding; this example uses **all
three**: Sentinel-2 L2A + Sentinel-1 GRD at 10 m GSD, plus Sentinel-3 OLCI (21
reflectance bands, 240 m) and SLSTR (6 reflectance + 3 brightness-temperature bands,
480/960 m). It exports with a lighter ~9.6 km ground cover than THOR's own reference
configuration, to keep the per-tile compute cost manageable. The THOR encoder runs
inside an openEO UDF as an **ONNX model**, so no PyTorch, TerraTorch or
`thor_terratorch_ext` is needed in the UDF sandbox — only `onnxruntime`.

The output is a task-agnostic embedding cube (embedding dimension depends on the THOR
model size exported — 192 for `thor_v1_tiny`) on the input pixel grid that can be
materialised to NetCDF and reused for land-cover classification, change detection,
similarity search, etc.

### Why ONNX instead of vendored PyTorch?

This repo has two patterns for running a foundation model inside a UDF: vendor the
PyTorch model directly (see [`../tessera`](../tessera/)) or export it once to ONNX (see
[`../terramind`](../terramind/)). THOR follows the ONNX pattern, for the same reason
TerraMind does, only more so:

- **THOR isn't a self-contained model file.** Unlike TESSERA's single vendorable
  `model.py`, THOR is only buildable through two separate git repositories
  (`FM4CS/THOR` + `FM4CS/thor_terratorch_ext`) plus `terratorch` plus `torch` —
  vendoring FlexiViT, 2D ALiBi position encoding, and the multi-modality
  patch-embedding/merge logic into a UDF sandbox would be a large, fragile surface
  compared to shipping one traced ONNX graph.
- **Sandbox dependencies stay minimal.** `onnxruntime` is a small, stable, pip-installable
  wheel that's easy to ship via `udf-dependency-archives`. Installing `torch` +
  `terratorch` + two custom git packages into an ephemeral UDF sandbox is heavier and
  more failure-prone.
- **The trade-off we accept:** THOR's headline "compute-adaptive resolution" feature
  (choosing `ground_cover`/`patch_sizes` at will) becomes a *design-time* choice baked
  into the exported graph rather than a *runtime* UDF parameter — see the note below and
  [Caveats](#caveats). For a pipeline that needs to switch resolution profiles per job
  without re-exporting, a vendored-PyTorch approach would be required instead; that's
  not this example's goal.

> **Note on THOR's compute-adaptive resolution.** THOR's `ground_cover`/`patch_sizes`
> knobs let you trade spatial detail for compute cost without retraining — but that
> choice is baked into the traced graph at ONNX export time, not switchable at
> inference. See [Caveats](#caveats) for what stays flexible after export and how to
> ship multiple resolution profiles.

> **Adding Sentinel-3 makes this heavier than S1+S2-only.** OLCI's 240 m native GSD
> needs enough `ground_cover` to span more than a couple of patches, which is why this
> example uses a 9.6 km tile (960x960 px at 10 m for S1/S2) rather than an S1+S2-only
> profile's much smaller tile — see [Caveats](#caveats) for the compute-cost trade-off,
> THOR's own (much heavier) reference config, and a smaller-scale alternative.

## Requirements

- An openEO backend with `SENTINEL2_L2A`, `SENTINEL1_GRD`, `SENTINEL3_OLCI_L1B` and
  `SENTINEL3_SLSTR` collections (this example targets the
  [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) openEO endpoint).
  **Verify the exact Sentinel-3 band names exposed by your backend** (e.g. via
  `connection.describe_collection("SENTINEL3_OLCI_L1B")`) — this example assumes
  `Oa01_radiance`..`Oa21_radiance` and `S1_radiance_an`..`S6_radiance_an` /
  `S7_BT_in`..`S9_BT_in`, matching the raw SAFE product band names, but naming can vary
  by provider — e.g. CDSE's `SENTINEL3_OLCI_L1B`/`SENTINEL3_SLSTR` actually expose plain
  `B01`..`B21` and `S1`..`S9` rather than the raw SAFE names, and this example's
  notebook/UDF are set up for that.
- A backend that supports the `udf-dependency-archives` job option (check with your
  backend provider).
- Offline: a Python environment with `thor`, `thor_terratorch_ext`, `terratorch`, `torch`
  and `onnx` for the one-off ONNX export (see
  [THOR](https://github.com/FM4CS/THOR) and
  [thor_terratorch_ext](https://github.com/FM4CS/thor_terratorch_ext)).

## Architecture

```
SENTINEL2_L2A      (B02, B03, B04, B08, SCL cloud mask, temporal median, DN -> reflectance) ─┐
SENTINEL1_GRD      (VV, VH sigma0-ellipsoid, temporal median, linear -> dB)                  │
SENTINEL3_OLCI_L1B (B01..B21 radiance, temporal median)                                      ├─ merge_cubes / resample ─► single 9.6 km tile ─► UDF
SENTINEL3_SLSTR    (S1..S6 radiance, S7..S9 BT passthrough, temporal median)                 ┘                                                     │
                                                                                                                                                     ▼
                                                                                                     ONNX THOR encoder (radiance -> reflectance via
                                                                                                     solar flux + SZA, merge_method="mean", per-group
                                                                                                     patch sizes, standardization baked in)
                                                                                                                                        │
                                                                                                                                        ▼
                                                                                              embedding cube (D-band, D = model embed dim)
                                                                                              on the input pixel grid
```

All bands are resampled to a common 10 m grid before merging: THOR's ONNX graph
internally resizes each channel to its own baked-in GSD/`ground_cover`-derived working
resolution regardless of the resolution the pixels arrive at, so the openEO side only
needs one shared grid, not per-sensor tiling.

## Two-step workflow

1. **Offline, once**: export a THOR encoder to ONNX with
   [`export_thor_to_onnx.py`](./export_thor_to_onnx.py). The script:
   - downloads the pretrained weights from Hugging Face (`FM4CS/THOR-1.0-*`) via
     `thor_terratorch_ext`,
   - builds a `thor_v1_{tiny,small,base,large}` backbone from 5 modality groups
     (`S1GRD`, `S2L2A`, `S3OLCI`, `S3SLSTR_REFL`, `S3SLSTR_BT`) with per-group
     `patch_sizes`, `select_patch_strategy="min"` and `merge_method="mean"` — a lighter
     ~9.6 km profile chosen over THOR's own (much heavier) multi-sensor reference example
     for manageable compute cost (see [Caveats](#caveats)),
   - bakes THOR's published per-band pretraining mean/std
     (`THOR_NORMALIZATION_PARAMS`) into the ONNX graph as constants, so the UDF hands
     the model physical values directly,
   - exports to `.onnx` and writes a `bands_order.json` sidecar recording the resolved
     input channel order (36 channels), ground cover, patch sizes and embedding
     dimension.

   Zip the `.onnx` file and `bands_order.json` together and upload the zip to somewhere
   the backend can reach over HTTPS.

2. **Online**: [`thor-embedding.ipynb`](./thor-embedding.ipynb) loads S1/S2/S3 via
   openEO, runs [`udf_thor_embedding.py`](./udf_thor_embedding.py) with the ONNX archive
   plus the shared `onnxruntime` archive supplied through `udf-dependency-archives`,
   saves the embedding cube as NetCDF, and sanity-checks it with an unsupervised K-means
   clustering.

## Why materialise the embedding as its own cube?

- **Reusable intermediate.** The embedding cube is task-agnostic. Once produced, the
  same cube can feed a land-cover classifier, a crop-type model, a change-detection
  routine, or a similarity-search index.

## Files

- `export_thor_to_onnx.py` — offline ONNX export helper (run cell-by-cell).
- `udf_thor_embedding.py` — the openEO UDF: loads the ONNX session, converts Sentinel-3
  OLCI/SLSTR radiance to the reflectance THOR expects (via a reimplemented
  `radiance_to_reflectance`/`compute_sza` — Sentinel-2 reflectance scaling and
  Sentinel-1 dB conversion happen upstream in the openEO graph instead, since they're
  simple scene-independent formulas), encodes a
  tile, upsamples the token grid back to the input pixel grid so
  `apply_neighborhood`'s pixel-index overlap trim keeps working.
- `thor-embedding.ipynb` — end-to-end example: load S2 L2A + S1 GRD + S3 OLCI/SLSTR, run
  the UDF, cluster the embedding with K-means, overlay on the S2 RGB.

## Caveats

- **THOR expects specific physical units per sensor**: Sentinel-2 L2A reflectance in
  `[0, 1]` (raw DN / 10000), Sentinel-1 GRD sigma0 in **dB** (not linear power — its SAR
  pretraining statistics are fit in log space), and Sentinel-3 OLCI/SLSTR TOA
  **reflectance** in `[0, 2]` (not the raw TOA radiance the L1B collections provide).
  Sentinel-2/Sentinel-1 conversion happens in the openEO graph (`thor-embedding.ipynb`);
  Sentinel-3 conversion still happens in the UDF, since it needs a per-scene solar
  zenith angle (see below). Get any of these wrong and the baked-in standardization
  will silently produce meaningless embeddings.
- **Sentinel-3 radiance -> reflectance requires a solar zenith angle (SZA)**, which
  needs per-pixel tie-point grids for full accuracy. This example instead uses a
  **single-point SZA approximation** (`center_lat`/`center_lon`/`acquisition_datetime`
  passed via UDF `context`, ~1° accuracy), matching the simplified approach
  `thor_terratorch_ext` itself documents for single scenes without tie-point data. For
  large AOIs or oblique viewing geometry this approximation degrades; use the tie-point
  grids in the raw SEN3 product if you need per-pixel accuracy.
- **THOR is single-timestep** in this example: all bands are reduced to a temporal
  median (or nearest available OLCI/SLSTR acquisition) before encoding. A BAP composite
  works equally well and can be swapped in for S1/S2.
- **THOR's compute-adaptive resolution is a per-export choice, not a runtime one.**
  This example uses a lighter ~9.6 km ground cover (960 px @ 10 m for the S1/S2 side)
  with per-modality patch sizes (`S1GRD`/`S2L2A`: 32 px, `S3OLCI`: 8 px, `S3SLSTR_REFL`:
  4 px, `S3SLSTR_BT`: 2 px, `select_patch_strategy="min"`) — chosen over THOR's own
  reference config (26 880 m ground cover, 2688 px @ 10 m, `S3OLCI`: 16 px,
  `S3SLSTR_REFL`: 8 px, `S3SLSTR_BT`: 4 px) for manageable compute cost; the reference
  config is kept as a commented-out alternative in `export_thor_to_onnx.py`. That
  trade-off is baked into the traced ONNX graph's weights and position encoding at
  export time — a single `.onnx` file cannot switch resolution at inference. To use a
  different profile (e.g. the heavier reference config, or dropping Sentinel-3 back
  out), re-run `export_thor_to_onnx.py` with different
  `GROUND_COVER`/`PATCH_SIZES`/`MODALITIES` values and ship the resulting archive as a
  separate model, selectable via the UDF's `onnx_dir`/`onnx_filename` context. What
  genuinely *is* dynamic at inference, because THOR internally resizes any input tile to
  its fixed working resolution before patch embedding: the exact pixel shape of the tile
  handed to the UDF (batch, height and width are all dynamic ONNX axes).
- **If you scale this example to a bigger AOI with `apply_neighborhood` chunking, the
  chunk size must match the exported model's working pixel size** (960x960 px for the
  default profile above) — **not** a small, generic chunk size like the 128x128/224x224
  px used in `../tessera`/`../terramind`. THOR's dynamic H/W axes exist so ragged edge
  tiles don't error, not so arbitrary chunk sizes work correctly: every tile handed to the
  UDF gets bilinear-resized internally to the fixed pixel size baked in at export,
  regardless of what it actually represents on the ground. Feed it a 128x128 px chunk
  (1.28 km at 10 m) against a model exported for a 9.6 km ground cover, and that chunk
  gets stretched ~7.5x to fill the grid the model was trained to interpret as 9.6 km —
  not a quality degradation, but spatially meaningless output. Use
  `apply_neighborhood(size=[{"dimension": "x", "value": 960, "unit": "px"}, {"dimension":
  "y", "value": 960, "unit": "px"}], overlap=[...])` (matching whatever `GROUND_COVER` you
  exported with) so each chunk maps 1:1 to what the model expects; openEO still handles
  the tiling/restitching across a larger AOI exactly like the smaller-chunk examples, just
  with a bigger per-tile footprint.
- **Compute cost is meaningfully higher than an S1+S2-only export**, though far lighter
  than THOR's own reference config. A 960x960 px, 36-channel float32 tile is roughly
  130 MB in memory per UDF invocation (vs. ~1 GB for the 2688x2688 px reference config)
  — this AOI is sized to be *one single tile* (`apply_neighborhood` size == the full
  extent, no spatial tiling), not something to run over a large area unmodified. For a
  lighter-weight experiment, drop `S3OLCI`/`S3SLSTR_*` from `MODALITIES` and shrink
  `GROUND_COVER` back down to get the original S1+S2-only profile.
- **Fine-tuning**: retrain / adapt with `terratorch fit` offline (see
  `thor_terratorch_ext`), re-export with the same script, point the UDF's
  `onnx_filename` at the new archive: nothing else changes.
