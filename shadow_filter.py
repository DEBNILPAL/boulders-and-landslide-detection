
# --- File: src/ml_models/shadow_filter.py ---
import numpy as np
from matplotlib.colors import LightSource

def compute_illumination_map(dtm_array, az=135, alt=45):
    ls = LightSource(azdeg=az, altdeg=alt)
    illumination = ls.shade(dtm_array, vert_exag=1, blend_mode='soft')
    return illumination


def get_shadow_mask(illumination_map, threshold=0.2):
    shadow_mask = (illumination_map < threshold).astype(np.uint8) * 255
    return shadow_mask
