"""openEO UDF: THOR patch embeddings via ONNX Runtime (S1 + S2 + S3 OLCI/SLSTR).

No PyTorch, no TerraTorch, this UDF only needs ``onnxruntime`` + ``numpy`` in
the sandbox. The THOR encoder is exported to ONNX **offline** with
``export_thor_to_onnx.py`` (https://github.com/FM4CS/THOR +
https://github.com/FM4CS/thor_terratorch_ext) and shipped to the backend as a
zip via the ``udf-dependency-archives`` job option (same mechanism as
``onnx_deps.zip`` in the terramind example). The archive must contain the
``.onnx`` graph and the ``bands_order.json`` sidecar written by the export
script.

Standardization is baked into the exported ONNX graph using THOR's published
pretraining mean/std per band, so this UDF only has to hand the model the
right *physical units*. Sentinel-2 reflectance scaling and Sentinel-1 dB
conversion are simple, scene-independent formulas, so they're done upstream
in the openEO graph (see ``thor-embedding.ipynb``) rather than here — this
UDF only has to convert Sentinel-3 OLCI/SLSTR, since that needs a per-scene
solar zenith angle:
    - Sentinel-2 L2A bands arrive already as reflectance in [0, 1],
    - Sentinel-1 GRD bands arrive already as sigma0 in **dB** (THOR's SAR
      stats are fit in log space, unlike TerraMind's linear-power convention),
    - Sentinel-3 OLCI/SLSTR reflectance bands are converted here from L1B TOA
      **radiance** to TOA reflectance in [0, 2] using THOR's own
      ``radiance = pi * L / (solar_flux * cos(SZA))`` formula, since that's
      the unit THOR's pretraining stats are fit in — see
      ``thor_terratorch_ext.datasets.sentinel3_utils``, reimplemented below so
      the sandbox doesn't need ``thor_terratorch_ext``/``torch``),
    - Sentinel-3 SLSTR brightness-temperature bands as Kelvin, unconverted.

Expected input cube bands (order irrelevant, resolved by name — **verify
these against your backend's actual collection band names**, e.g. via
``connection.describe_collection(...)``; naming can vary by provider):
    S2L2A:        B02 (BLUE), B03 (GREEN), B04 (RED), B08 (NIR) — reflectance [0, 1]
    S1GRD:        VV, VH — sigma0-ellipsoid in dB (converted upstream in the graph)
    SENTINEL3_OLCI_L1B: B01 .. B21 — TOA radiance (CDSE naming; verify against
                        your backend, naming varies). Prefixed "OLCI_" in the
                        input cube to avoid colliding with S2's B02/B03/B04/B08.
    SENTINEL3_SLSTR:    S1 .. S6 — TOA radiance (CDSE naming)
                        S7, S8, S9 — brightness temperature (K) (CDSE naming)
                        Prefixed "SLSTR_" in the input cube for the same reason.

Context:
    onnx_dir      : folder alias the archive extracts to (default: "thor_onnx")
    onnx_filename : .onnx filename inside that folder
                    (default: "thor_v1_base_encoder.onnx")
    bands_json    : bands_order.json filename inside that folder
                    (default: "bands_order.json")
    center_lat, center_lon : AOI centroid in decimal degrees, used for a
                    single-point solar zenith angle approximation (required
                    for OLCI/SLSTR bands; not needed for S1/S2-only cubes).
    acquisition_datetime : ISO 8601 UTC timestamp of the Sentinel-3 overpass
                    used for the same SZA approximation, e.g.
                    "2018-07-15T10:30:00".
"""
import functools
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from openeo.metadata import CubeMetadata
from openeo.udf import XarrayDataCube

# onnxruntime is supplied via the ``udf-dependency-archives`` job option that
# extracts to ./onnx_deps (same convention as ../terramind).
sys.path.append("onnx_deps")
import onnxruntime as ort


# Mean solar spectral irradiance (mW m^-2 nm^-1), tabulated by THOR for OLCI
# bands Oa01-Oa21 and SLSTR reflectance bands S1-S6. Copied verbatim from
# thor_terratorch_ext.datasets.sentinel3_utils (MIT licensed) so this UDF
# doesn't need to import thor_terratorch_ext/torch in the ONNX-only sandbox.
OLCI_SOLAR_FLUX = [
    1500.96, 1672.84, 1849.84, 1893.60, 1879.35, 1758.86, 1614.05, 1497.63,
    1463.22, 1438.04, 1373.91, 1239.14, 1221.16, 1213.50, 1202.85, 1149.11,
    938.91, 911.24, 876.70, 808.90, 684.75,
]
SLSTR_SOLAR_FLUX = [1798.94, 1488.56, 936.94, 358.14, 240.20, 75.82]

OLCI_BANDS = [f"Oa{i:02d}" for i in range(1, 22)]           # THOR: S3:Oa01_reflectance..Oa21
SLSTR_REFL_BANDS = [f"S{i}" for i in range(1, 7)]             # THOR: S3:S1_reflectance_an..S6
SLSTR_BT_BANDS = [f"S{i}" for i in range(7, 10)]              # THOR: S3:S7_BT_in..S9_BT_in


def _identity(x, **_):
    return np.asarray(x, dtype=np.float32)   # already in THOR's expected unit, converted upstream in the openEO graph


def _compute_sza_deg(lat_deg: float, lon_deg: float, dt: datetime) -> float:
    """Single-point solar zenith angle approximation (~1 deg accuracy).

    Reimplements thor_terratorch_ext.datasets.sentinel3_utils.compute_sza
    """
    doy = dt.timetuple().tm_yday
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
    utc_h = dt.hour + dt.minute / 60 + dt.second / 3600
    ha = math.radians(15.0 * (utc_h - 12.0 + lon_deg / 15.0))
    lat_r = math.radians(lat_deg)
    cos_sza = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.acos(max(0.001, min(1.0, cos_sza))))


def _radiance_to_reflectance(radiance, solar_flux, cos_sza):
    cos_sza = np.clip(cos_sza, 0.01, 1.0)
    refl = (np.pi * radiance.astype(np.float32)) / (solar_flux * cos_sza)
    return np.clip(refl, 0.0, 2.0).astype(np.float32)


def _s3_refl(x, *, solar_flux, cos_sza):
    return _radiance_to_reflectance(x, solar_flux, cos_sza)


def _s3_bt(x, **_):
    return np.asarray(x, dtype=np.float32)   # already Kelvin, no conversion needed


# Map THOR's internal band names to (openeo_band_name, convert_fn, extra kwargs).
THOR_BAND_SOURCE = {
    "S2:Blue": ("B02", _identity, {}),
    "S2:Green": ("B03", _identity, {}),
    "S2:Red": ("B04", _identity, {}),
    "S2:NIR": ("B08", _identity, {}),
    "S1:IW-VV": ("VV", _identity, {}),
    "S1:IW-VH": ("VH", _identity, {}),
}
for _i, _b in enumerate(OLCI_BANDS):
    THOR_BAND_SOURCE[f"S3:{_b}_reflectance"] = (
        f"OLCI_B{_i + 1:02d}", _s3_refl, {"solar_flux": OLCI_SOLAR_FLUX[_i]}
    )
for _i, _b in enumerate(SLSTR_REFL_BANDS):
    THOR_BAND_SOURCE[f"S3:{_b}_reflectance_an"] = (
        f"SLSTR_{_b}", _s3_refl, {"solar_flux": SLSTR_SOLAR_FLUX[_i]}
    )
for _b in SLSTR_BT_BANDS:
    THOR_BAND_SOURCE[f"S3:{_b}_BT_in"] = (f"SLSTR_{_b}", _s3_bt, {})

DEFAULT_ONNX_DIR = "thor_onnx"
DEFAULT_ONNX_FILENAME = "thor_v1_base_encoder.onnx"
DEFAULT_BANDS_JSON = "bands_order.json"


@functools.lru_cache(maxsize=2)
def _load_session(onnx_path: str) -> ort.InferenceSession:
    if not Path(onnx_path).exists():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}. Make sure the THOR ONNX zip "
            f"is listed in the 'udf-dependency-archives' job option and that "
            f"context['onnx_dir']/context['onnx_filename'] match its layout."
        )
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 2
    return ort.InferenceSession(onnx_path, sess_options=so,
                                 providers=["CPUExecutionProvider"])


def _resolve_dependency_file(base_dir: str, filename: str) -> str:
    """Resolve a file shipped via udf-dependency-archives.

    Accept both layouts:
    - <alias>/<filename>
    - <alias>/<nested-folder>/<filename>

    This keeps the UDF tolerant to whether the zip contains files directly at
    its root or under an extra package folder like torch_v1_base_onnx/.
    """
    base = Path(base_dir)
    direct = base / filename
    if direct.exists():
        return direct.as_posix()

    search_root = base if base.exists() else base.parent
    if search_root.exists():
        matches = sorted(search_root.rglob(filename))
        if matches:
            return matches[0].as_posix()

    raise FileNotFoundError(
        f"Could not find {filename} under dependency archive path {base_dir!r}. "
        f"Checked direct path {direct.as_posix()} and recursive search under "
        f"{search_root.as_posix() if search_root.exists() else base.parent.as_posix()}."
    )


@functools.lru_cache(maxsize=2)
def _load_band_order(json_path: str) -> tuple:
    with open(json_path) as f:
        cfg = json.load(f)
    return tuple(cfg["band_order"])


def _build_input(arr: xr.DataArray, band_order, cos_sza: float) -> np.ndarray:
    avail = list(arr.coords["bands"].values.tolist())
    channels = []
    for thor_band in band_order:
        if thor_band not in THOR_BAND_SOURCE:
            raise ValueError(f"No openEO source mapping known for THOR band {thor_band!r}.")
        openeo_band, convert, kwargs = THOR_BAND_SOURCE[thor_band]
        if openeo_band not in avail:
            raise ValueError(f"Missing band {openeo_band!r} in input cube. Available: {avail}")
        raw = arr.sel(bands=openeo_band).values.astype(np.float32)
        channels.append(convert(raw, cos_sza=cos_sza, **kwargs))
    return np.nan_to_num(np.stack(channels, axis=0), nan=0.0).astype(np.float32)   # (C, H, W)


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    context = context or {}
    onnx_dir = context.get("onnx_dir", DEFAULT_ONNX_DIR)
    onnx_filename = context.get("onnx_filename", DEFAULT_ONNX_FILENAME)
    bands_json = context.get("bands_json", DEFAULT_BANDS_JSON)
    onnx_path = _resolve_dependency_file(onnx_dir, onnx_filename)
    bands_json_path = _resolve_dependency_file(onnx_dir, bands_json)

    arr = cube.get_array()   # (bands, y, x) or (t, bands, y, x)
    if "t" in arr.dims:
        arr = arr.median(dim="t", skipna=True)

    band_order = _load_band_order(bands_json_path)

    # Single-point SZA approximation for any S3 OLCI/SLSTR reflectance bands
    # in band_order. Not needed (and not required in context) for S1/S2-only
    # exports.
    needs_sza = any(b.startswith("S3:") and "reflectance" in b for b in band_order)
    cos_sza = 1.0
    if needs_sza:
        lat = context["center_lat"]
        lon = context["center_lon"]
        dt = datetime.fromisoformat(context["acquisition_datetime"])
        cos_sza = math.cos(math.radians(_compute_sza_deg(lat, lon, dt)))

    sess = _load_session(onnx_path)

    x = _build_input(arr, band_order, cos_sza)   # (C, H, W)
    in_x = arr.coords["x"].values
    in_y = arr.coords["y"].values
    H, W = x.shape[-2:]

    feats = sess.run(None, {"x": x[np.newaxis, ...]})[0]   # (1, D, h, w) token grid
    feats = feats[0].astype(np.float32)
    emb_dim, tok_h, tok_w = feats.shape

    # Upsample (nearest-neighbour repeat) back to the input chunk's own pixel
    # grid/coordinates, rather than returning a coarser grid with custom
    # coordinates. apply_neighborhood expects a UDF's output to align with
    # the grid it was given; returning a mismatched resolution/extent forces
    # the backend to reproject each chunk's odd little tile back into its
    # expected layer grid (geotrellis's RasterRegionReproject), which was
    # causing executor JVM heap OOMs — not a Python-side memory problem.
    # The common 40 m input grid keeps this repeat cheap (e.g. 240x240 px
    # rather than 960x960), so it's fine to do unconditionally.
    factor_y = H // tok_h
    factor_x = W // tok_w
    feats = np.repeat(np.repeat(feats, factor_y, axis=1), factor_x, axis=2)
    feats = feats[:, :H, :W]

    out = xr.DataArray(
        feats,
        dims=["bands", "y", "x"],
        coords={
            "bands": [f"emb_{i:03d}" for i in range(emb_dim)],
            "y": in_y,
            "x": in_x,
        },
    )
    return XarrayDataCube(out)


def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    context = context or {}
    onnx_dir = context.get("onnx_dir", DEFAULT_ONNX_DIR)
    bands_json = context.get("bands_json", DEFAULT_BANDS_JSON)
    bands_json_path = _resolve_dependency_file(onnx_dir, bands_json)
    with open(bands_json_path) as f:
        embed_dim = json.load(f)["embed_dim"]
    return metadata.rename_labels(
        dimension="bands",
        target=[f"emb_{i:03d}" for i in range(embed_dim)],
    )
