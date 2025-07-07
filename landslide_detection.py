import cv2
import numpy as np

def detect_landslides(image, slope_map, slope_threshold=30):
    """
    Detect landslide regions by combining slope map and edge detection.
    Returns list of filtered contours.
    """
    # Resize slope map to match image shape
    slope_map_resized = cv2.resize(slope_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Binary mask of steep slope areas
    #slope_mask = (slope_map_resized > slope_threshold).astype(np.uint8) * 255
    slope_mask = cv2.normalize(slope_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    slope_mask = (slope_mask > slope_threshold).astype(np.uint8) * 255

    # Edge detection to find terrain boundaries
    edges = cv2.Canny(image, 50, 150)

    # Combine slope and edge info
    combined_mask = cv2.bitwise_and(edges, slope_mask)

    # Apply morphological closing to connect broken edges
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours from closed mask
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    return filtered_contours
