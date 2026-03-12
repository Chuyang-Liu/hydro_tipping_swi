from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin


def recharge_to_m_per_yr(arr: np.ndarray, units: str = 'mm/yr') -> np.ndarray:
    return (arr / 1000.0) if units.lower().startswith('mm') else arr


def detect_lat_lon_names(ds: xr.Dataset, da: xr.DataArray) -> Tuple[str, str]:
    cand_lat = ['lat', 'latitude', 'y']
    cand_lon = ['lon', 'longitude', 'x']
    lat_name = next((n for n in cand_lat if n in da.coords), None) or next((n for n in cand_lat if n in ds.coords), None)
    lon_name = next((n for n in cand_lon if n in da.coords), None) or next((n for n in cand_lon if n in ds.coords), None)
    if not lat_name or not lon_name:
        raise ValueError(f'Could not detect lat/lon coords. DataArray coords={list(da.coords)}, Dataset coords={list(ds.coords)}')
    return lat_name, lon_name


def nc_latlon_to_geotiff(
    nc_path: str | Path,
    out_tif: str | Path,
    var_name: Optional[str] = None,
    units: str = 'mm/yr',
    compress: str = 'deflate',
) -> Path:
    nc_path, out_tif = Path(nc_path), Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(nc_path) as ds:
        if var_name is None:
            data_vars = [k for k in ds.data_vars.keys()]
            if len(data_vars) != 1:
                raise ValueError(f'Please provide var_name. Found variables: {data_vars}')
            var_name = data_vars[0]
        da = ds[var_name]
        lat_name, lon_name = detect_lat_lon_names(ds, da)
        lat = np.asarray(da[lat_name].values)
        lon = np.asarray(da[lon_name].values)
        arr = np.asarray(da.values, dtype='float32')
        arr = recharge_to_m_per_yr(arr, units=units).astype('float32')

        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError('Expected 1D lat/lon coordinates.')
        xres = float(np.abs(np.diff(lon)).mean())
        yres = float(np.abs(np.diff(lat)).mean())
        west = float(lon.min() - xres / 2.0)
        north = float(lat.max() + yres / 2.0)
        if lat[0] < lat[-1]:
            arr = np.flipud(arr)
        transform = from_origin(west, north, xres, yres)
        profile = {
            'driver': 'GTiff', 'height': arr.shape[0], 'width': arr.shape[1], 'count': 1,
            'dtype': 'float32', 'crs': 'EPSG:4326', 'transform': transform, 'compress': compress,
            'nodata': np.nan,
        }
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(arr, 1)
    return out_tif
