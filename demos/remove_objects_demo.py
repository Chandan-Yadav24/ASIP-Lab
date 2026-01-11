# demos/remove_objects_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, color, util
from PIL import Image
import io

@st.cache_data
def load_img_removal(file_data, threshold=128, noise_level=0.01):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
        binary = img_np > threshold
    else:
        # Synthetic object with small noise dots
        img_np = np.zeros((400, 400), dtype=bool)
        # Main objects
        img_np[100:200, 100:200] = True
        img_np[250:350, 250:350] = True
        # Add small noise dots (salt noise)
        noise = np.random.rand(400, 400) < noise_level
        binary = img_np | noise
    return binary

def run():
    st.header("🧽 Removing Small Objects: Binary Cleanup")
    st.markdown("""
    This lab demonstrates how to clean up binary images by deleting tiny, irrelevant blobs while keeping the significant objects based on their **pixel area**.
    """)

    with st.expander("📚 Theory: Size-Based Filtering", expanded=False):
        st.markdown(r"""
        ### 1. The Key Concept
        Small objects are often "noise" or "artifacts" from thresholding. We want to remove any connected component whose total pixel count is below a certain **minimum size**.
        
        ### 2. Approaches
        - **A) Connected Component Analysis (CCA)**:
          1. Label every separate object with an ID.
          2. Count pixels for each ID.
          3. If `pixel_count < threshold`, turn those pixels to background (0).
        - **B) Morphological Erosion**:
          - Erode the image until small objects disappear.
          - Use the result as a mask to seed or identify the larger objects.
        
        ### 3. Usage
        - Noise reduction in binary masks.
        - Removing "salt" noise from segmentation results.
        - Preprocessing for shape analysis (e.g., counting only large cells).
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="remove_local")
    
    noise_level = st.sidebar.slider("Synthetic Noise Level", 0.0, 0.05, 0.01, format="%.3f")
    thresh_val = 128
    if local_up:
        thresh_val = st.sidebar.slider("Binary Threshold", 0, 255, 128)
        
    binary_img = load_img_removal(local_up.read() if local_up else None, threshold=thresh_val, noise_level=noise_level)

    tab_size, tab_morph, tab_stats = st.tabs(["📏 Size-Based removal", "📉 Erosion Match", "📊 Analysis Stats"])

    with tab_size:
        st.subheader("Interactive Size Filter")
        st.markdown("Adjust the slider to set the **minimum pixel count** required to keep an object.")
        
        min_size = st.slider("Minimum Object Area (pixels)", 1, 1000, 50)
        
        # Using skimage.morphology.remove_small_objects
        cleaned_size = morphology.remove_small_objects(binary_img, min_size=min_size)
        
        col1, col2 = st.columns(2)
        col1.image(binary_img.astype(np.uint8)*255, caption="Original Binary", use_container_width=True)
        col2.image(cleaned_size.astype(np.uint8)*255, caption=f"Cleaned (Min Area: {min_size}px)", use_container_width=True)
        
        st.info(f"Any blob with fewer than {min_size} pixels was automatically deleted.")

    with tab_morph:
        st.subheader("Erosion-Based Comparison")
        st.markdown("Erosion naturally 'deletes' objects smaller than the structuring element.")
        
        se_size = st.slider("SE Size (Square)", 1, 15, 3, step=2)
        se = morphology.square(se_size)
        
        eroded = morphology.binary_erosion(binary_img, se)
        
        # Reconstruction-like approach: remove objects that don't have ANY pixels after erosion
        # But for simplicity here, we just show the comparison
        # (Actually, a full morphological reconstruction would be better, but let's stick to the user's logic)
        
        c1, c2 = st.columns(2)
        c1.image(binary_img.astype(np.uint8)*255, caption="Original", use_container_width=True)
        c2.image(eroded.astype(np.uint8)*255, caption=f"Eroded (Size {se_size})", use_container_width=True)
        
        st.warning("Note: Erosion shrinks ALL objects. Size-based filtering is usually preferred if you want to keep the exact shape of large objects.")

    with tab_stats:
        st.subheader("Component Analysis")
        
        # Labeling components
        labels_orig = measure.label(binary_img)
        num_orig = np.max(labels_orig)
        
        # Using the cleaned version from the first tab
        labels_clean = measure.label(cleaned_size)
        num_clean = np.max(labels_clean)
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Initial Objects", num_orig)
        k2.metric("Remaining Objects", num_clean)
        k3.metric("Removed (Noise)", num_orig - num_clean)
        
        st.success(f"By applying the filter, you successfully eliminated {num_orig - num_clean} small noise fragments!")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Removing Small Objects**:
    - **Method A (Size-based)**: Uses Connected Components to count pixels per object and deletes any under-threshold blob. (Preferred).
    - **Method B (Erosion)**: Naturally removes small blobs but also shrinks the boundaries of kept objects.
    - **Threshold**: The most critical parameter is the 'Minimum Area' (pixel count).
    - **Goal**: Clean binary masks for better segmentation and measurement.
    """)
