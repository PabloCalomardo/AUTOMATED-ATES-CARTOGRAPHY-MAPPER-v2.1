#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from pathlib import Path


def _outside_nodata_mask(valid_mask):
	"""Return nodata cells connected to image border (outside DEM footprint)."""

	import numpy as np

	invalid = ~valid_mask
	rows, cols = invalid.shape
	outside = np.zeros_like(invalid, dtype=bool)
	q = deque()

	for c in range(cols):
		if invalid[0, c] and not outside[0, c]:
			outside[0, c] = True
			q.append((0, c))
		if invalid[rows - 1, c] and not outside[rows - 1, c]:
			outside[rows - 1, c] = True
			q.append((rows - 1, c))

	for r in range(rows):
		if invalid[r, 0] and not outside[r, 0]:
			outside[r, 0] = True
			q.append((r, 0))
		if invalid[r, cols - 1] and not outside[r, cols - 1]:
			outside[r, cols - 1] = True
			q.append((r, cols - 1))

	while q:
		r, c = q.popleft()
		for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
			if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
				continue
			if invalid[rr, cc] and not outside[rr, cc]:
				outside[rr, cc] = True
				q.append((rr, cc))

	return outside


def _default_nodata_for_dtype(dtype_name: str):
	import numpy as np

	dtype = np.dtype(dtype_name)
	if np.issubdtype(dtype, np.floating):
		return np.nan
	return 0


def align_forest_to_dem(
	in_forest: str | Path,
	ref_dem: str | Path,
	out_forest: str | Path,
	forest_crs: str | None = None,
) -> Path:
	"""Align a forest raster to DEM grid (resolution, size, transform and CRS).

	Nearest-neighbour resampling is used because forest layers are commonly
	categorical/discrete. The output is also clipped to the DEM valid footprint.
	"""

	in_path = Path(in_forest).expanduser().resolve()
	ref_path = Path(ref_dem).expanduser().resolve()
	out_path = Path(out_forest).expanduser().resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)

	import numpy as np
	import rasterio
	from rasterio.enums import Resampling
	from rasterio.warp import reproject

	with rasterio.open(ref_path) as ref, rasterio.open(in_path) as src:
		dem_valid = ~ref.read(1, masked=True).mask
		src_crs = src.crs
		if src_crs is None and forest_crs:
			src_crs = rasterio.crs.CRS.from_user_input(forest_crs)
		out_nodata = src.nodata if src.nodata is not None else _default_nodata_for_dtype(src.dtypes[0])

		same_grid = (
			src.width == ref.width
			and src.height == ref.height
			and src.transform == ref.transform
			and src.crs == ref.crs
		)

		if src_crs != ref.crs and (src_crs is None or ref.crs is None):
			raise ValueError(
				"Forest and DEM CRS differ or are incomplete. Provide a forest CRS with --forest-crs "
				"(for example EPSG:25833), or use inputs with valid CRS metadata."
			)

		profile = src.profile.copy()
		profile.update(
			width=ref.width,
			height=ref.height,
			transform=ref.transform,
			crs=ref.crs,
			nodata=out_nodata,
			compress="deflate",
		)

		with rasterio.open(out_path, "w", **profile) as dst:
			for band_idx in range(1, src.count + 1):
				if same_grid:
					dst_arr = src.read(band_idx)
				else:
					src_arr = src.read(band_idx)
					dst_arr = np.empty((ref.height, ref.width), dtype=src_arr.dtype)
					reproject(
						source=src_arr,
						destination=dst_arr,
						src_transform=src.transform,
						src_crs=src_crs,
						dst_transform=ref.transform,
						dst_crs=ref.crs,
						src_nodata=src.nodata,
						dst_nodata=out_nodata,
						resampling=Resampling.nearest,
					)

				dst_arr[~dem_valid] = out_nodata
				dst.write(dst_arr, band_idx)

	return out_path


def fill_dem_simple(in_dem: str | Path, out_dem: str | Path) -> Path:
	"""Fill DEM nodata with a simple inpainting.

	Uses `rasterio.fill.fillnodata` when nodata pixels exist.
	If nodata is missing or fill is unavailable, the DEM is copied as-is.

	Returns the output path.
	"""

	in_path = Path(in_dem).expanduser().resolve()
	out_path = Path(out_dem).expanduser().resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)

	import numpy as np
	import rasterio

	with rasterio.open(in_path) as src:
		profile = src.profile.copy()
		band1 = src.read(1, masked=True)
		nodata = src.nodata

	# If there is no explicit nodata or no missing pixels, just copy.
	if nodata is None:
		with rasterio.open(in_path) as src, rasterio.open(out_path, "w", **profile) as dst:
			dst.write(src.read())
		return out_path

	mask = getattr(band1, "mask", False)
	if mask is False:
		with rasterio.open(in_path) as src, rasterio.open(out_path, "w", **profile) as dst:
			dst.write(src.read())
		return out_path
	try:
		if hasattr(mask, "any") and not mask.any():
			with rasterio.open(in_path) as src, rasterio.open(out_path, "w", **profile) as dst:
				dst.write(src.read())
			return out_path
	except Exception:
		# If mask behavior is unexpected, do a safe copy.
		with rasterio.open(in_path) as src, rasterio.open(out_path, "w", **profile) as dst:
			dst.write(src.read())
		return out_path

	# We have nodata; attempt fill.
	arr = band1.filled(nodata).astype("float32")
	valid_mask = ~band1.mask  # True where valid
	outside_invalid = _outside_nodata_mask(valid_mask=valid_mask)
	fill_support_mask = np.logical_or(valid_mask, outside_invalid).astype("uint8")

	try:
		from rasterio.fill import fillnodata

		filled = fillnodata(
			arr,
			mask=fill_support_mask,
			max_search_distance=100.0,
			smoothing_iterations=0,
		)
		out_arr = filled.astype("float32")
	except Exception:
		out_arr = band1.filled(nodata)

	# Keep DEM footprint exactly as input (no square frame growth after fill).
	out_arr[outside_invalid] = nodata

	profile.update(count=1, compress="deflate", dtype=str(out_arr.dtype), nodata=nodata)
	with rasterio.open(out_path, "w", **profile) as dst:
		dst.write(out_arr, 1)

	return out_path


def normalize_forest_for_flowpy(
	in_forest: str | Path,
	out_forest: str | Path,
) -> Path:
	"""Normalize forest raster for Flow-Py to a 0..1 range.

	Valid values are clipped to be non-negative and scaled by the maximum value
	found in the layer (0..max). Nodata and DEM-outside footprint remain nodata.
	"""

	in_path = Path(in_forest).expanduser().resolve()
	out_path = Path(out_forest).expanduser().resolve()
	out_path.parent.mkdir(parents=True, exist_ok=True)

	import numpy as np
	import rasterio

	with rasterio.open(in_path) as src:
		arr = src.read(1)
		profile = src.profile.copy()
		nodata = src.nodata

	arr_f = arr.astype("float32", copy=True)
	if nodata is None:
		valid = np.isfinite(arr_f)
	else:
		if np.isnan(nodata):
			valid = np.isfinite(arr_f)
		else:
			valid = arr != nodata

	norm = np.zeros_like(arr_f, dtype="float32")
	valid_nonneg = np.clip(arr_f[valid], 0.0, None)
	max_valid = float(np.max(valid_nonneg)) if valid_nonneg.size else 0.0
	if max_valid > 0.0:
		norm[valid] = valid_nonneg / max_valid
	else:
		norm[valid] = 0.0

	out_nodata = -9999.0
	norm[~valid] = out_nodata

	profile.update(count=1, dtype="float32", nodata=out_nodata, compress="deflate")
	with rasterio.open(out_path, "w", **profile) as dst:
		dst.write(norm, 1)

	return out_path

