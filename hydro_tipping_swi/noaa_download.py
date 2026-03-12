from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

ROOT_INDEX = "https://coast.noaa.gov/slrdata/"
BASE_DIR = ROOT_INDEX


def requote(url: str) -> str:
    return requests.utils.requote_uri(url)


def get_region_codes(session: requests.Session, root_index: str = ROOT_INDEX) -> list[str]:
    r = session.get(root_index, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    codes: set[str] = set()
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        text = (a.get_text() or "").strip()
        for candidate in (text, href):
            m = re.fullmatch(r"([A-Z]{2})(?:/|/index\.html)?", candidate)
            if m:
                codes.add(m.group(1))
    return sorted(codes)


def try_fetch_urllist(code: str, session: requests.Session, base_dir: str = BASE_DIR) -> list[str] | None:
    urllist_url = urljoin(base_dir, f"{code}/URLlist_{code}.txt")
    r = session.get(urllist_url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    urls = [u.strip() for u in r.text.split() if u.strip().startswith("http")]
    return urls or None


def scrape_region_files(code: str, session: requests.Session, base_dir: str = BASE_DIR) -> list[str]:
    region_index = urljoin(base_dir, f"{code}/index.html")
    r = session.get(region_index, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    urls: list[str] = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if href and re.search(r"\.(zip|gpkg)\b", href, flags=re.IGNORECASE):
            urls.append(requote(urljoin(region_index, href)))

    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def filename_from_url(url: str) -> str:
    name = os.path.basename(urlparse(url).path)
    return unquote(name) if name else "download.bin"


def download_one(url: str, out_path: str | Path, session: requests.Session, retries: int = 3, chunk_mb: int = 1) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skipped"

    last_err = None
    for _attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", "0")) or None
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                chunk_size = chunk_mb * 1024 * 1024
                pbar = tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name, leave=False) if tqdm is not None else None
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if pbar is not None:
                                pbar.update(len(chunk))
                if pbar is not None:
                    pbar.close()
                os.replace(tmp, out_path)
                return "downloaded"
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(1)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def download_noaa_slr_vectors(
    out_dir: str | Path = "./noaa_slr_vectors",
    regions: Optional[Iterable[str]] = None,
    only: str = "all",
    delay: float = 0.25,
    save_url_manifest: bool = True,
    session: Optional[requests.Session] = None,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()

    wanted = {"all", "zip", "gpkg"}
    if only not in wanted:
        raise ValueError(f"only must be one of {wanted}")

    region_codes = list(regions) if regions is not None else get_region_codes(session)
    downloaded: list[Path] = []

    for code in region_codes:
        urls = try_fetch_urllist(code, session) or scrape_region_files(code, session)
        if only == "zip":
            urls = [u for u in urls if u.lower().endswith('.zip')]
        elif only == "gpkg":
            urls = [u for u in urls if u.lower().endswith('.gpkg')]
        region_dir = out_dir / code
        region_dir.mkdir(parents=True, exist_ok=True)
        if save_url_manifest:
            (region_dir / f"URLlist_{code}.txt").write_text("\n".join(urls) + "\n")
        for url in urls:
            out_path = region_dir / filename_from_url(url)
            download_one(url, out_path, session=session)
            downloaded.append(out_path)
            if delay > 0:
                time.sleep(delay)
    return downloaded
