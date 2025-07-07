# --- File: src/6_generate_report.py ---
import csv

def save_boulder_data(boulders, filename):
    """
    Save boulder coordinates and sizes to a CSV file.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y", "Diameter"])
        for x, y, d in boulders:
            writer.writerow([x, y, d])

def save_landslide_data(contours, filename):
    """
    Save landslide contour point data to a CSV file.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Contour_ID", "Points"])
        for i, c in enumerate(contours):
            points = [(p[0][0], p[0][1]) for p in c]
            writer.writerow([i, points])


def save_landslide_slopes(slope_stats, output_csv):
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Landslide_ID", "Avg_Slope", "Min_Slope", "Max_Slope"])
        for i, (avg, min_s, max_s) in enumerate(slope_stats):
            writer.writerow([i, avg, min_s, max_s])