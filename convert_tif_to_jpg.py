from PIL import Image
import os
import glob

# === INPUT FOLDER ===
input_folder = "masks"     # Replace with your .tif folder name
output_folder = "images"    # New folder for .jpg files

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through .tif files and convert
for tif_path in glob.glob(os.path.join(input_folder, "*.tif")):
    try:
        img = Image.open(tif_path)
        # Convert to grayscale (if needed) and save as .jpg
        jpg_path = os.path.join(output_folder, os.path.splitext(os.path.basename(tif_path))[0] + ".jpg")
        img.convert("L").save(jpg_path, "JPEG")
        print(f"✅ Converted: {tif_path} → {jpg_path}")
    except Exception as e:
        print(f"❌ Error converting {tif_path}: {e}")
