# demos/skeleton_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage import morphology, util
import io

@st.cache_data
def load_img_skeleton(file_data, threshold=128, test_case="Synthetic"):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
        binary = img_np > threshold
    else:
        if test_case == "Synthetic":
            # Simple shapes: Circle and Rectangle
            img_np = np.zeros((300, 300), dtype=bool)
            img_np[50:150, 50:150] = True
            rr, cc = np.ogrid[:300, :300]
            mask = (rr - 200)**2 + (cc - 200)**2 <= 60**2
            img_np[mask] = True
            binary = img_np
        elif test_case == "OCR":
            # Draw text
            img = Image.new('L', (300, 150), 0)
            draw = ImageDraw.Draw(img)
            # Use default font, Draw 'A' and '8'
            draw.text((20, 10), "A", fill=255, size=100)
            draw.text((150, 10), "8", fill=255, size=100)
            # Make it thicker for better skeletonization demo
            img_np = np.array(img)
            binary = img_np > 50
            binary = morphology.binary_dilation(binary, morphology.square(7))
        else: # Vessel
            # Create a branching structure
            img_np = np.zeros((300, 300), dtype=bool)
            img_np[50:250, 145:155] = True # Trunk
            img_np[100:150, 150:250] = True # Right branch
            img_np[150:200, 50:150] = True # Left branch
            # Thicken
            binary = morphology.binary_dilation(img_np, morphology.disk(8))
    return binary

def run():
    st.header("🦴 Skeletonizing: Extracting the Core")
    st.markdown("""
    Skeletonization (or Thinning) reduces thick foreground objects to a **stick-figure representation** only 1 pixel wide, while preserving the original topology and connectivity.
    """)

    with st.expander("📚 Theory: Geometric Peeling", expanded=False):
        st.markdown(r"""
        ### 1. Intuition
        Imagine "peeling" the outer layers of an object like an onion. You continue until only a "spine" remains, where removing any more pixels would break the object apart.
        
        ### 2. Properties
        - **Connectivity**: If two parts were connected, they stay connected in the skeleton.
        - **Topology**: The number of holes and branches is preserved.
        - **Thinness**: The final result is ideally 1 pixel thick everywhere.
        
        ### 3. Skeletonization vs Erosion
        - **Erosion**: Shrinks everything; can make small objects disappear or break thin parts.
        - **Skeletonization**: Intelligent shrinking; stops when it hits the "medial axis" to preserve structure.
        
        ### 4. Applications
        - **OCR**: Analyzing the "skeleton" of a letter to recognize it regardless of font thickness.
        - **Medical**: Extracting centerlines of blood vessels or neurons for length/branching analysis.
        - **Fingerprints**: Thinning ridges to detect minutiae (junctions and endings).
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    test_case = st.sidebar.selectbox("Test Case", ["Synthetic", "OCR", "Vessel Branching"])
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="skel_local")
    
    thresh_val = 128
    if local_up:
        thresh_val = st.sidebar.slider("Binary Threshold", 0, 255, 128)
        
    binary_img = load_img_skeleton(local_up.read() if local_up else None, threshold=thresh_val, test_case=test_case)

    tab_demo, tab_ocr, tab_compare = st.tabs(["✨ Skeleton Lab", "🔠 OCR & Medical", "🆚 Skel vs Erosion"])

    with tab_demo:
        st.subheader("Medial Axis Extraction")
        
        skeleton = morphology.skeletonize(binary_img)
        
        col1, col2 = st.columns(2)
        col1.image(binary_img.astype(np.uint8)*255, caption="Original Binary", use_container_width=True)
        # Overlay for better visibility
        overlay = binary_img.astype(float) * 0.3
        overlay[skeleton] = 1.0
        col2.image(overlay, caption="Skeleton Overlay (Medial Axis)", use_container_width=True)
        
        st.info("The skeleton perfectly follows the center line of the shapes.")

    with tab_ocr:
        st.subheader("Structure Analysis")
        st.markdown("Thinning reveals the 'core' representation of complex structures.")
        
        # Use thinning for a different visual/algorithm comparison if desired, 
        # but skeletonize is the standard in skimage
        skel = morphology.skeletonize(binary_img)
        
        c1, c2 = st.columns(2)
        c1.image(binary_img.astype(np.uint8)*255, caption="Input Structure", use_container_width=True)
        c2.image(skel.astype(np.uint8)*255, caption="Structural Skeleton", use_container_width=True)
        
        st.success("Regardless of how 'bold' the font or 'thick' the vessel is, the skeleton looks the same.")

    with tab_compare:
        st.subheader("Why Simple Erosion Fails")
        st.markdown("Erosion cannot be used for thinning because it destroys connectivity.")
        
        iter_count = st.slider("Erosion Iterations", 1, 30, 10)
        eroded = binary_img.copy()
        for _ in range(iter_count):
            eroded = morphology.binary_erosion(eroded)
            
        skel = morphology.skeletonize(binary_img)
        
        e1, e2 = st.columns(2)
        e1.image(eroded.astype(np.uint8)*255, caption=f"Eroded ({iter_count} iter)", use_container_width=True)
        e2.image(skel.astype(np.uint8)*255, caption="Skeletonized (Topology Preserved)", use_container_width=True)
        
        st.warning("Note how erosion makes parts disappear or break, while skeletonization keeps the 'skeleton' connected!")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Skeletonization**:
    - **Goal**: Reduce foreground to a 1-pixel wide 'stick figure'.
    - **Preservation**: Must keep **Connectivity** and **Topology** (holes/branches).
    - **Method**: Iterative 'peeling' of boundary pixels.
    - **Key Application**: Medial Axis extraction, character recognition, blood vessel analysis.
    - **Distinction**: Unlike erosion, it never breaks or removes parts that are essential to the shape's connectivity.
    """)
