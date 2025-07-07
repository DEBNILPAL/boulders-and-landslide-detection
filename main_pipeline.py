# --- File: src/9_main_pipeline.py ---
import os
import cv2
import argparse
import numpy as np
from preprocess_images import preprocess_image
from elevation_analysis import compute_slope_aspect
from landslide_detection import detect_landslides
from boulder_detection import detect_boulders
from visualization import draw_landslides, draw_boulders
from generate_report import save_boulder_data, save_landslide_data
from utils import find_matching_dtm
from generate_report import save_landslide_slopes

# ML models
from ml_models.yolo_detect import run_yolo_detection
from ml_models.unet_inference import run_unet_segmentation
from ml_models.shadow_filter import get_shadow_mask
from ml_models.hillshade_slope import compute_hillshade


def compute_landslide_slopes(contours, slope_map):
    landslide_slopes = []
    for contour in contours:
        slopes = []
        for point in contour:
            x, y = point[0]
            if 0 <= y < slope_map.shape[0] and 0 <= x < slope_map.shape[1]:
                slopes.append(slope_map[int(y)][int(x)])
        if slopes:
            avg_slope = round(np.mean(slopes), 2)
            max_slope = round(np.max(slopes), 2)
            min_slope = round(np.min(slopes), 2)
            landslide_slopes.append((avg_slope, min_slope, max_slope))
        else:
            landslide_slopes.append((None, None, None))
    return landslide_slopes

def run_pipeline(image_path, dtm_path, output_prefix, method="traditional"):
    # Step 1: Preprocess Image
    image = preprocess_image(image_path)

    # Step 2: Compute Slope and Aspect
    slope, aspect = compute_slope_aspect(dtm_path)

    # Step 3: Detect Landslides
    # contours = detect_landslides(image, slope)   approach taken before ML models..

    # After Ml models
    if method.lower() == "unet":
        landslide_mask = run_unet_segmentation(image_path)
        contours, _ = cv2.findContours(landslide_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours = detect_landslides(image, slope)

    # Step 4: Detect Boulders
    # boulders = detect_boulders(image)     approach taken before ML models..

    # After Ml models
    if method.lower() == "yolo":
        boulder_map, boulders = run_yolo_detection(image_path)
    else:
        boulders = detect_boulders(image)


    # Step 5: Visualization
    landslide_map = draw_landslides(image, contours)
    boulder_map = draw_boulders(landslide_map, boulders)

    # Save annotated image
    os.makedirs("outputs/annotated_maps", exist_ok=True)
    os.makedirs("outputs/csv", exist_ok=True)
    annotated_output = f'outputs/annotated_maps/{output_prefix}_annotated.png'
    cv2.imwrite(annotated_output, boulder_map)

    # Step 6: Save Reports
    save_boulder_data(boulders, f'outputs/csv/{output_prefix}_boulders.csv')
    save_landslide_data(contours, f'outputs/csv/{output_prefix}_landslides.csv')

    # Step 7: Slope Info for Landslides
    slope_stats = compute_landslide_slopes(contours, slope)
    for i, (avg, min_s, max_s) in enumerate(slope_stats):
        print(f"Landslide {i}: Avg = {avg}°, Min = {min_s}°, Max = {max_s}°")

    # Step 8: Save slope stats to CSV
    save_landslide_slopes(slope_stats, f'outputs/csv/{output_prefix}_slopes.csv')

    print(f"Pipeline completed. Output saved as: {annotated_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to TMC/OHRC image")
    parser.add_argument("--dtm", required=False, help="Path to DTM image")
    parser.add_argument("--output", required=True, help="Output file prefix")
    parser.add_argument("--method", required=False, choices=["traditional", "unet", "yolo"], default="traditional")

    args = parser.parse_args()

    # Try to auto-find DTM if not given
    dtm_path = args.dtm
    if not dtm_path:
        dtm_path = find_matching_dtm(args.image, "../data/dtm/")
        if not dtm_path:
            raise FileNotFoundError("DTM not provided and no match found.")

    run_pipeline(args.image, dtm_path, args.output, args.method)



# IF You know the DTM then write this .
# python src/main_pipeline.py --image data/tmc_images/PIA13998~orig.jpg --dtm data/dtm/sample_dtm.tif --output test1

# IF You Don't know the DTM then write this .
# python src/main_pipeline.py --image data/tmc_images/PIA13998~orig.jpg --output tycho
# It will auto-match and proceed ✅

# 1. Traditional CV
# python src/main_pipeline.py --image path.jpg --dtm path.tif --output result1 --method traditional

# 2. U-Net based landslide detection
# python src/main_pipeline.py --image path.jpg --dtm path.tif --output result2 --method unet

# 3. YOLO-based boulder detection
# python src/main_pipeline.py --image path.jpg --dtm path.tif --output result3 --method yolo
