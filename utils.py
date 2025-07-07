# --- File: src/11_utils.py ---

#import os

#def find_matching_dtm(image_filename, dtm_folder):
#    """
#    Find a likely DTM file that matches a given image based on filename pattern.
#    Returns the full path to the matched DTM file or None.
#    """
#    base = os.path.splitext(os.path.basename(image_filename))[0]
#    candidates = [f for f in os.listdir(dtm_folder) if base in f and f.endswith('.tif')]
#    if candidates:
#        return os.path.join(dtm_folder, candidates[0])
#    else:
#        print(f"❌ No matching DTM found for: {image_filename}")
#        return None



# --- File: src/11_utils.py ---
import os
import rasterio
from PIL import Image


def find_matching_dtm(image_path, dtm_folder):
    # Load input image size
    image = Image.open(image_path)
    image_size = image.size  # (width, height)

    best_match = None
    min_diff = float('inf')

    for filename in os.listdir(dtm_folder):
        if filename.endswith(".tif"):
            dtm_path = os.path.join(dtm_folder, filename)
            try:
                with rasterio.open(dtm_path) as dtm:
                    dtm_size = (dtm.width, dtm.height)
                    diff = abs(dtm_size[0] - image_size[0]) + abs(dtm_size[1] - image_size[1])
                    if diff < min_diff:
                        min_diff = diff
                        best_match = dtm_path
            except Exception as e:
                print(f"Warning: Couldn't read {filename}: {e}")

    return best_match

