import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin

# Use relative path from src/ folder
image = cv2.imread("../data/tmc_images/0003869~large.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image not found. Check the path.")

h, w = image.shape
dummy_elevation = np.ones((h, w), dtype=np.float32) * 1000

with rasterio.open(
    "../data/dtm/dummy_dtm.tif", 'w',
    driver='GTiff', height=h, width=w,
    count=1, dtype='float32', crs='+proj=latlong',
    transform=from_origin(0, 0, 1, 1)
) as dst:
    dst.write(dummy_elevation, 1)

print("✅ Dummy DTM created at: data/dtm/dummy_dtm.tif")
