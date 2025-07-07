
# --- File: src/ml_models/hillshade_slope.py ---
import numpy as np
from matplotlib.colors import LightSource

def compute_slope(dem):
    dzdx = np.gradient(dem, axis=1)
    dzdy = np.gradient(dem, axis=0)
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return slope


def compute_hillshade(dem, az=315, alt=45):
    ls = LightSource(azdeg=az, altdeg=alt)
    hillshade = ls.hillshade(dem, vert_exag=1)
    return hillshade
