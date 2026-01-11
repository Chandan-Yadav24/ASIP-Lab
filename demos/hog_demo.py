# demos/hog_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color, exposure, data
from PIL import Image
import io

@st.cache_data
def load_img_hog(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Astronaut or astronaut's head is classic for edge/feature demos
        img_np = data.astronaut()
        img_np = color.rgb2gray(img_np)
    return img_np

def run():
    st.header("🐗 Histogram of Oriented Gradients (HOG)")
    st.markdown("""
    HOG is a powerful feature descriptor used for object detection. It represents an object's appearance by summarizing the distribution of **edge directions** in local patches.
    """)

    with st.expander("📚 Theory: The Directional Map", expanded=False):
        st.markdown(r"""
        ### 1. The Core Idea
        Objects have characteristic edge patterns. HOG captures these by:
        1. Computing **Gradients** ($I_x, I_y$) to find edge strength and direction.
        2. Dividing the image into small **Cells** (e.g., 8x8 pixels).
        3. Computing a **Histogram** of orientations for each cell.
        4. **Normalizing** groups of cells (Blocks) to be robust to illumination changes.
        
        ### 2. The Detection Pipeline
        HOG is often combined with a **Support Vector Machine (SVM)** to detect:
        - Pedestrians
        - Faces
        - Vehicles
        
        ### ❄️ The Frosted Window Analogy
        Imagine viewing a building through many small frosted windows. You can't see details, but you can record the **direction of strong lines** (vertical, horizontal). That map is distinctive enough to recognize the building even if lighting changes!
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="hog_local")
    img = load_img_hog(local_up.read() if local_up else None)
    
    # Rescale intensity for better visibility (ensuring float 0-1 range)
    img_rescaled = exposure.rescale_intensity(img, out_range=(0, 1))

    st.sidebar.markdown("### 🧰 Parameters")
    orientations = st.sidebar.slider("Bins (Orientations)", 4, 18, 9)
    ppc = st.sidebar.slider("Pixels per Cell", 4, 32, 8)
    cpb = st.sidebar.slider("Cells per Block", 1, 4, 2)

    tab_vis, tab_normalize, tab_table = st.tabs(["👁️ HOG Visualizer", "🌓 Illumination Robustness", "📊 Summary Table"])

    with tab_vis:
        st.subheader("Visualizing HOG Features")
        st.markdown(f"Current Config: {orientations} bins, {ppc}x{ppc} cell, {cpb}x{cpb} block.")
        
        # Compute HOG
        fd, hog_image = feature.hog(img, orientations=orientations, 
                                    pixels_per_cell=(ppc, ppc),
                                    cells_per_block=(cpb, cpb), 
                                    visualize=True)
        
        # Rescale HOG image for display (ensuring safe float 0-1 range)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 1))
        
        col1, col2 = st.columns(2)
        col1.image(img_rescaled, caption="Original Image", use_container_width=True)
        col2.image(hog_image_rescaled, caption="HOG Descriptor Visualization", use_container_width=True)
        
        st.info(f"Feature Vector Length: **{len(fd)}** numbers represent this image.")
        st.caption("Each line in the visualization represents the dominant gradient direction in that cell.")

    with tab_normalize:
        st.subheader("Why Normalization Matters")
        st.markdown("HOG is robust to lighting because it normalizes intensity within blocks.")
        
        dark_img = img * 0.5 # Simulate low light
        bright_img = np.clip(img * 1.5, 0, 1) # Simulate high contrast
        
        # Visualization logic
        fd_dark = feature.hog(dark_img, orientations=orientations, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb))
        fd_bright = feature.hog(bright_img, orientations=orientations, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb))
        
        # Compare a small portion of the descriptor vectors
        st.write("First 10 values of the normalized descriptor:")
        st.line_chart({"Dark Image": fd_dark[:10], "Bright Image": fd_bright[:10]})
        
        st.success("Notice how the descriptor values are nearly identical despite the huge difference in raw image brightness!")

    with tab_table:
        st.subheader("HOG Component Summary")
        st.table({
            "Part": ["Gradients", "Cells", "Orientation Histograms", "Block Normalization", "Feature Vector"],
            "What it does": [
                "Compute edge strength/direction",
                "Divide into local regions",
                "Count directions per cell",
                "Group cells & normalize",
                "Concatenate all blocks"
            ],
            "Why it matters": [
                "Captures shape",
                "Preserves local patterns",
                "Encodes local edge layout",
                "Robust to illumination",
                "Input for Classifiers (SVM)"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **HOG**: Histogram of Oriented Gradients.
    - **Main Use**: Rigid object detection (Pedestrians, Faces).
    - **Key Steps**: Centered gradients $\rightarrow$ Binning $\rightarrow$ Block Normalization.
    - **Invariance**: Robust to lighting/contrast; moderately robust to small translations.
    - **Classifier**: Typically paired with Linear SVM.
    """)
