from PIL import Image
import os

def convert_jpg_to_compressed_tif(input_path, output_path=None, max_width=2000, compression="tiff_lzw"):
    # Open the original image
    img = Image.open(input_path)

    # Resize if larger than max_width (preserve aspect ratio)
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.ANTIALIAS)
        print(f"🔧 Resized to: {new_size}")

    # Convert to 8-bit RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Set output path
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + "_compressed.tif"

    # Save with LZW compression
    img.save(output_path, format="TIFF", compression=compression)
    print(f"✅ Saved compressed TIF to: {output_path}")

# Example usage
convert_jpg_to_compressed_tif("image.jpg")
