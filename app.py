# --- File: gui/app.py ---
import streamlit as st
from PIL import Image
import os
import uuid
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main_pipeline import run_pipeline
from src.utils import find_matching_dtm

st.set_page_config(page_title="🌕 Lunar Landslide & Boulder Detector", layout="wide")

main_title = "BHARATIYA ANTARIKSH"
subtitle = "HACKATHON 2025"
org = "INDIAN SPACE RESEARCH ORGANISATION (ISRO)"
flag_url = "https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg"

# --- Animated Galaxy Background with Floating Astronaut ---
st.markdown(f"""
    <style>
    .hero-section {{
        background-image: url("https://i.postimg.cc/LXP4XGdR/moon-5.jpg");
        background-size: cover;
        height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        position: relative;
        text-align: center;
        padding-top: 60px;
    }}

    .title {{
        font-size: 64px;
        font-weight: bold;
        color: #ffa500;
        text-shadow: 0 0 10px #00ccff, 0 0 20px #39ff14;
        animation: glow 2s ease-in-out infinite alternate;
    }}

    .subtitle {{
        font-size: 20px;
        margin-top: 10px;
        color: #000080;
    }}

    @keyframes glow {{
        from {{
            text-shadow: 0 0 10px #00ccff, 0 0 20px #39ff14;
        }}
        to {{
            text-shadow: 0 0 20px #00ccff, 0 0 30px #39ff14;
        }}
    }}
    </style>

    <div class="hero-section">
        <div class="title">{main_title}<br>{subtitle}</div>
        <div class="subtitle">Organised by<br><b>{org}</b></div>
        <img src="{flag_url}" width="80px" />
    </div>
""", unsafe_allow_html=True)

# --- File Uploader ---
st.subheader("🗉 Upload TMC/OHRC Lunar Image (.jpg)")
image_file = st.file_uploader("Upload .jpg Image", type=["jpg", "jpeg"])

st.subheader("🌐 Optional: Upload Matching DTM (.tif)")
dtm_file = st.file_uploader("Upload .tif DTM File (optional)", type=["tif"])

# --- Detection Method ---
method = st.selectbox("🔍 Choose Detection Method", ["traditional", "unet", "yolo"])

# --- Run Button ---
if image_file:
    unique_id = str(uuid.uuid4())[:8]
    input_path = f"gui/temp_image_{unique_id}.jpg"
    dtm_path = None

    with open(input_path, "wb") as f:
        f.write(image_file.getbuffer())

    if dtm_file:
        dtm_path = f"gui/temp_dtm_{unique_id}.tif"
        with open(dtm_path, "wb") as f:
            f.write(dtm_file.getbuffer())
    else:
        dtm_path = find_matching_dtm(input_path, "data/dtm/")
        if not dtm_path:
            st.error("⚠️ DTM not found and not uploaded manually.")
            st.stop()

    st.success("✅ Files uploaded. Running detection...")
    run_pipeline(input_path, dtm_path, unique_id, method)

    # --- Show Output Image ---
    output_img_path = f"outputs/annotated_maps/{unique_id}_annotated.png"
    st.image(output_img_path, caption="🚀 Annotated Image", use_column_width=True)

    # --- Download links for CSVs ---
    st.markdown("### 📅 Download CSV Outputs")
    st.download_button("Download Boulders CSV", open(f"outputs/csv/{unique_id}_boulders.csv", "rb").read(), file_name="boulders.csv")
    st.download_button("Download Landslides CSV", open(f"outputs/csv/{unique_id}_landslides.csv", "rb").read(), file_name="landslides.csv")
    st.download_button("Download Slope Stats CSV", open(f"outputs/csv/{unique_id}_slopes.csv", "rb").read(), file_name="slopes.csv")
