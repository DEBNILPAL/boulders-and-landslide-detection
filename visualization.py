# --- File: src/5_visualization.py ---
import cv2

def draw_landslides(image, contours):
    """
    Draw detected landslide contours on the image.
    """
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out

def draw_boulders(image, boulders):
    """
    Draw detected boulders as red circles on the image.
    """
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    for (x, y, r) in boulders:
        cv2.circle(out, (x, y), r // 2, (0, 0, 255), 2)
    return out
