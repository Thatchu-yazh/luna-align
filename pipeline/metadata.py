import io
import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.io import MemoryFile
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

def extract_geospatial_metadata(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    meta = {
        "filename": uploaded_file.name,
        "is_geotiff": False,
        "width": None,
        "height": None,
        "crs": "Local Orbital Frame (Non-Projected)",
        "gsd_est": "Undefined (Raw Optical)",
        "bounds": None
    }
    if RASTERIO_AVAILABLE:
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    meta["is_geotiff"] = bool(src.crs)
                    meta["width"] = src.width
                    meta["height"] = src.height
                    meta["crs"] = str(src.crs) if src.crs else "Unprojected Lunar Raster"
                    meta["bounds"] = src.bounds
                    t = src.transform
                    if t and t[0] != 0.0 and abs(t[0]) != 1.0:
                        meta["gsd_est"] = f"{abs(t[0]):.3f} m/px"
        except Exception:
            pass
    if meta["width"] is None:
        img = Image.open(io.BytesIO(file_bytes))
        meta["width"], meta["height"] = img.size
    return meta