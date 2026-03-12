from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import fiona
import numpy as np
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from shapely.geometry import shape as shp_shape
from shapely.ops import transform as shp_transform, unary_union


def bounds_intersect(b1, b2) -> bool:
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def to_2d(geom):
    try:
        return shp_transform(lambda x, y, z=None: (x, y), geom)
    except Exception:
        return geom


def make_valid(geom):
    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
    except Exception:
        return None
    try:
        from shapely.make_valid import make_valid as _make_valid
        geom = _make_valid(geom)
    except Exception:
        geom = geom.buffer(0)
    return None if geom is None or geom.is_empty else geom


def state_from_shp_name(shp_path: str | Path) -> Optional[str]:
    left = Path(shp_path).stem.split('__', 1)[0]
    st = left.split('_', 1)[0].upper()
    return st if re.fullmatch(r'[A-Z]{2}', st) else None


def collect_state_shapefiles(state: str, scenario_dirs: Iterable[str | Path], include_low: bool = False) -> list[Path]:
    state = state.upper()
    out: list[Path] = []
    for sd in map(Path, scenario_dirs):
        for shp in sd.glob('*.shp'):
            nm = shp.name.lower()
            if (not include_low) and ('_low_' in nm):
                continue
            if state_from_shp_name(shp) == state:
                out.append(shp)
    return sorted(out)


def read_shapes_from_shps(shp_list: Iterable[str | Path], target_crs: CRS):
    from pyproj import Transformer
    shapes = []
    minx = miny = math.inf
    maxx = maxy = -math.inf
    for shp in map(Path, shp_list):
        with fiona.open(shp) as src:
            src_crs = CRS.from_user_input(src.crs) if src.crs else None
            if src_crs is None:
                raise ValueError(f'{shp} has no CRS.')
            transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True) if src_crs != target_crs else None
            for feat in src:
                g = feat.get('geometry')
                if not g:
                    continue
                geom = to_2d(shp_shape(g))
                if transformer is not None:
                    geom = shp_transform(transformer.transform, geom)
                geom = make_valid(geom)
                if geom is None:
                    continue
                shapes.append(geom)
                b = geom.bounds
                minx, miny = min(minx, b[0]), min(miny, b[1])
                maxx, maxy = max(maxx, b[2]), max(maxy, b[3])
    if not shapes:
        raise ValueError('No valid shapes found.')
    return shapes, (minx, miny, maxx, maxy)


def ksat_cmhr_to_perm_m2(ksat_cmhr: np.ndarray) -> np.ndarray:
    return np.asarray(ksat_cmhr, dtype='float64') * PERM_FACTOR


def effective_perm_from_depth_arrays(depth_arrays: Dict[str, np.ndarray], nodata: float = POLARIS_NODATA) -> np.ndarray:
    keys = [k for k in POLARIS_DEPTHS_CM if k in depth_arrays]
    if not keys:
        raise ValueError('No recognized POLARIS depth arrays supplied.')
    total_thick = sum(POLARIS_DEPTHS_CM[k] for k in keys)
    out = None
    weight_sum = None
    for k in keys:
        arr = np.asarray(depth_arrays[k], dtype='float64')
        valid = np.isfinite(arr) & (arr != nodata)
        w = POLARIS_DEPTHS_CM[k] / total_thick
        contrib = np.where(valid, arr * w, 0.0)
        out = contrib if out is None else out + contrib
        weight_sum = np.where(valid, w, 0.0) if weight_sum is None else weight_sum + np.where(valid, w, 0.0)
    return np.where(weight_sum > 0, out / weight_sum, np.nan)


def mask_raster_to_geometry(src_path: str | Path, geometry, out_path: str | Path, crop: bool = True) -> Path:
    from rasterio.mask import mask
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        arr, transform = mask(src, [geometry], crop=crop, filled=True)
        profile = src.profile.copy()
        profile.update(height=arr.shape[1], width=arr.shape[2], transform=transform)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(arr)
    return out_path


def build_effective_perm_raster(
    polaris_paths: Dict[str, str | Path],
    aoi_shapes: Iterable,
    out_path: str | Path,
    nodata: float = -9999.0,
    compress: str = 'deflate',
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_path = Path(next(iter(polaris_paths.values())))
    with rasterio.open(sample_path) as src0:
        profile = src0.profile.copy()
        profile.update(dtype='float32', nodata=nodata, compress=compress)
        shape_mask = features.rasterize([(unary_union(list(aoi_shapes)), 1)], out_shape=(src0.height, src0.width), transform=src0.transform, fill=0, dtype='uint8')

        arrays = {}
        for depth, path in polaris_paths.items():
            with rasterio.open(path) as src:
                arr = src.read(1).astype('float64')
                if src.transform != src0.transform or src.crs != src0.crs or src.width != src0.width or src.height != src0.height:
                    dest = np.full((src0.height, src0.width), src.nodata if src.nodata is not None else np.nan, dtype='float32')
                    rasterio.warp.reproject(
                        source=arr, destination=dest,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=src0.transform, dst_crs=src0.crs,
                        resampling=Resampling.bilinear,
                    )
                    arr = dest.astype('float64')
                arrays[depth] = ksat_cmhr_to_perm_m2(arr)

        eff = effective_perm_from_depth_arrays(arrays, nodata=POLARIS_NODATA).astype('float32')
        eff[shape_mask == 0] = nodata
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(eff, 1)
    return out_path
