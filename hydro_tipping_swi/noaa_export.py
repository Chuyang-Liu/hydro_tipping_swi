from __future__ import annotations

import hashlib
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd

from .io_utils import load_json, save_json_atomic, slugify


def sanitize(text: str) -> str:
    return slugify(text)


def task_hash(*parts: object) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode('utf-8'))
        h.update(b'|')
    return h.hexdigest()[:12]


def delete_shp_family(shp_path: str | Path) -> None:
    shp_path = Path(shp_path)
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.fix']:
        p = shp_path.with_suffix(ext)
        if p.exists():
            p.unlink()


def write_meta(path: str | Path, obj: dict) -> Path:
    return save_json_atomic(path, obj)


def read_meta(path: str | Path) -> Optional[dict]:
    return load_json(path)


def zip_members(path: str | Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


def zip_contains(path: str | Path, suffix: str) -> bool:
    suffix = suffix.lower()
    return any(name.lower().endswith(suffix) for name in zip_members(path))


def unzip_if_needed(path: str | Path, out_dir: str | Path) -> Path:
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != '.zip':
        return path
    target = out_dir / path.stem
    if not target.exists():
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
    return target


def list_layers(vector_path: str | Path) -> list[str]:
    return list(gpd.list_layers(vector_path)['name'])


def layer_matches_any(layer_name: str, scenario: str, include_low: bool = False) -> bool:
    ln = str(layer_name).lower()
    if '__layer_list_failed__' in ln:
        return False
    if not include_low and '_low_' in ln:
        return False
    if '_slr_' not in ln:
        return False

    s = scenario.lower()
    if f'_slr_{s}' in ln:
        return True
    if include_low and f'_low_{s}' in ln:
        return True

    m = re.match(r'^(\d+)_0ft$', s)
    if m:
        n = m.group(1)
        if f'_slr_{n}ft' in ln:
            return True
        if include_low and f'_low_{n}ft' in ln:
            return True

    if s == '5_0ft':
        if ('_slr_5ft' in ln) or ('_slr_5_0ft' in ln) or ('_slr_5.0ft' in ln):
            return True
        if include_low and (('_low_5ft' in ln) or ('_low_5_0ft' in ln) or ('_low_5.0ft' in ln)):
            return True
    return False


def export_layer_to_shapefile(vector_path: str | Path, layer_name: str, out_shp: str | Path, target_crs: Optional[str] = None) -> Path:
    out_shp = Path(out_shp)
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_shp.with_name(out_shp.stem + '_tmp.shp')
    delete_shp_family(tmp)
    gdf = gpd.read_file(vector_path, layer=layer_name)
    if target_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(target_crs)
    gdf.to_file(tmp)
    delete_shp_family(out_shp)
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        src = tmp.with_suffix(ext)
        if src.exists():
            os.replace(src, out_shp.with_suffix(ext))
    return out_shp


def shp_complete_strict(shp_path: str | Path) -> bool:
    shp_path = Path(shp_path)
    need = [shp_path.with_suffix(ext) for ext in ['.shp', '.shx', '.dbf', '.prj']]
    return all(p.exists() and p.stat().st_size > 0 for p in need)


def export_matching_layers(
    vector_path: str | Path,
    scenario: str,
    out_dir: str | Path,
    include_low: bool = False,
    target_crs: Optional[str] = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for layer in list_layers(vector_path):
        if layer_matches_any(layer, scenario=scenario, include_low=include_low):
            out_shp = out_dir / f"{sanitize(Path(vector_path).stem)}__{sanitize(layer)}.shp"
            export_layer_to_shapefile(vector_path, layer, out_shp, target_crs=target_crs)
            outputs.append(out_shp)
    return outputs
