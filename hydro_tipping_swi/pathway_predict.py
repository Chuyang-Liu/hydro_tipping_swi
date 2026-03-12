from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .pathway_labels import CLASS_TO_ID_4

OUT_DTYPE = 'uint8'
OUT_NODATA = 0


def count_class_ids_in_uint8_raster(raster_path: str | Path) -> dict[int, int]:
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
    vals, counts = np.unique(arr[arr != OUT_NODATA], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def get_pixel_area_km2(src: rasterio.io.DatasetReader) -> float:
    a = abs(src.transform.a)
    e = abs(src.transform.e)
    if src.crs and src.crs.is_projected:
        return (a * e) / 1e6
    raise ValueError('Pixel area requires projected CRS for constant km^2 per pixel.')


def deploy_state_gate2stage_fast(
    perm_raster: str | Path,
    recharge_raster: str | Path,
    model_bundle: dict,
    out_raster: str | Path,
    slr_m: float,
    k_tip: float = 1.58e-12,
    perm_nodata: Optional[float] = None,
    recharge_resampling: Resampling = Resampling.nearest,
) -> Path:
    model = model_bundle['model']
    feature_cols = model_bundle['feature_cols']
    out_raster = Path(out_raster)
    out_raster.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(perm_raster) as perm_src:
        profile = perm_src.profile.copy()
        profile.update(dtype=OUT_DTYPE, nodata=OUT_NODATA, compress='deflate', predictor=2)
        with rasterio.open(out_raster, 'w', **profile) as dst:
            with rasterio.open(recharge_raster) as rech_src:
                with WarpedVRT(rech_src, crs=perm_src.crs, transform=perm_src.transform, width=perm_src.width, height=perm_src.height, resampling=recharge_resampling) as rech_vrt:
                    for _, win in perm_src.block_windows(1):
                        K = perm_src.read(1, window=win).astype('float64')
                        rech = rech_vrt.read(1, window=win).astype('float64')
                        out = np.zeros(K.shape, dtype=np.uint8)
                        nodata_mask = ~np.isfinite(K)
                        if perm_nodata is None:
                            if perm_src.nodata is not None:
                                nodata_mask |= (K == perm_src.nodata)
                        else:
                            nodata_mask |= (K == perm_nodata)

                        gate_mask = (~nodata_mask) & (K < k_tip)
                        out[gate_mask] = CLASS_TO_ID_4['SWI not accelerated']

                        pred_mask = (~nodata_mask) & np.isfinite(rech) & (K >= k_tip) & (K > 0)
                        if np.any(pred_mask):
                            feats = {
                                'log_permeability_m2': np.log10(K[pred_mask]),
                                'recharge_eff_m_per_yr': rech[pred_mask],
                                'slr_m': np.full(np.count_nonzero(pred_mask), slr_m, dtype='float64'),
                            }
                            X = np.column_stack([feats[c] for c in feature_cols])
                            pred = model.predict(X)
                            cls = np.zeros(np.count_nonzero(pred_mask), dtype=np.uint8)
                            cls[pred == 'Lateral-dominated'] = CLASS_TO_ID_4['Lateral-dominated']
                            cls[pred == 'Mixed'] = CLASS_TO_ID_4['Mixed']
                            cls[pred == 'Vertical-dominated'] = CLASS_TO_ID_4['Vertical-dominated']
                            out[pred_mask] = cls

                        out[nodata_mask] = OUT_NODATA
                        dst.write(out, 1, window=win)
    return out_raster
