# --- File: src/2_elevation_analysis.py ---
import numpy as np
import rasterio

def compute_slope_aspect(dtm_path):
    """
    Compute slope and aspect from the given DTM (Digital Terrain Model).
    Returns slope in degrees and aspect in degrees.
    """
    with rasterio.open(dtm_path) as dataset:
        elevation = dataset.read(1)
        transform = dataset.transform

    # Compute gradients in x and y directions
    dz_dx = np.gradient(elevation, axis=1)
    dz_dy = np.gradient(elevation, axis=0)

    # Calculate slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # Calculate aspect in degrees
    aspect = np.degrees(np.arctan2(-dz_dy, dz_dx))
    aspect = (aspect + 360) % 360  # Normalize to [0, 360]

    return slope, aspect