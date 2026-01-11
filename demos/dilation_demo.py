# demos/dilation_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology
import io

@st.cache_data
def load_img_dilation(file_data, threshold=128):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
        binary = img_np > threshold
    else:
        # Synthetic shapes for dilation demo
        img_np = np.zeros((400, 400), dtype=bool)
        # Broken line (Gap Bridging test)
        img_np[50:150, 50:52] = True
        img_np[170:270, 50:52] = True
        # Object with internal holes (Hole Filling test)
        img_np[100:300, 150:350] = True
        img_np[150:170, 200:220] = False
        img_np[230:250, 280:300] = False
        # Small fragments
        img_np[350:360, 50:60] = True
        binary = img_np
    return binary

def get_se(shape, size):
    if shape == "Square":
        return morphology.square(size)
    elif shape == "Disk":
        return morphology.disk(size // 2 if size > 1 else 1)
    else: # Cross
        return morphology.star(size // 2 if size > 1 else 1)

def run():
    st.header("📈 Morphological Dilation: Growing & Bridging")
    st.markdown("""
    Dilation is the dual of erosion. It **expands objects**, fills small internal holes, and bridges narrow gaps between disconnected segments.
    """)

    with st.expander("📚 Theory: What is Dilation?", expanded=False):
        st.markdown(r"""
        ### 1. Intuition
        Think of sliding a template (Structuring Element) over the image. 
        - If the template touches **AT LEAST ONE** foreground pixel, the center pixel becomes white.
        - Result: Objects grow outward.
        
        ### 2. Binary Definition
        $$A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$$
        Where $A$ is the image, $B$ is the Structuring Element (SE), and $\hat{B}$ is its reflection.
        
        ### 3. Key Effects
        1. **Thickens Objects**: Boundaries move outward.
        2. **Fills Holes**: Small dark spots inside white objects disappear.
        3. **Bridges Gaps**: Nearby disconnected parts merge into one.
        4. **Duality**: Dilation of foreground $\leftrightarrow$ Erosion of background.
        
        ### 4. Grayscale Dilation
        In grayscale images, dilation acts as a **MAX Filter**. Every pixel is replaced by the highest intensity in its neighborhood. This expands bright patches and lightens the image.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="dilation_local")
    
    thresh_val = 128
    if local_up:
        thresh_val = st.sidebar.slider("Binary Threshold", 0, 255, 128)
        
    binary_img = load_img_dilation(local_up.read() if local_up else None, threshold=thresh_val)
    
    st.sidebar.markdown("### 🧰 Structuring Element (SE)")
    se_shape = st.sidebar.selectbox("Shape", ["Square", "Disk", "Cross"])
    se_size = st.sidebar.slider("Size", 1, 15, 3, step=2)
    se = get_se(se_shape, se_size)

    tab_binary, tab_bridge, tab_gray = st.tabs(["🔳 Growth Lab", "🌉 Bridging & Filling", "🌫️ Grayscale (Max Filter)"])

    with tab_binary:
        st.subheader("Object Growth Visualizer")
        
        iterations = st.slider("Successive Dilations", 1, 5, 1)
        
        dilated = binary_img.copy()
        for _ in range(iterations):
            dilated = morphology.binary_dilation(dilated, se)
            
        col1, col2 = st.columns(2)
        # Convert booleans to uint8 * 255 for proper display
        col1.image(binary_img.astype(np.uint8) * 255, caption="Original Binary", use_container_width=True)
        col2.image(dilated.astype(np.uint8) * 255, caption=f"Dilated ({iterations} iter, size={se_size})", use_container_width=True)
        
        st.info("Notice how objects become thicker and smaller fragments grow into larger blobs.")

    with tab_bridge:
        st.subheader("Bridging Gaps & Filling Holes")
        
        # We'll use a larger SE to show bridging
        se_bridge = morphology.square(7)
        dilated_bridge = morphology.binary_dilation(binary_img, se_bridge)
        
        b1, b2 = st.columns(2)
        b1.image(binary_img.astype(np.uint8) * 255, caption="Broken Segments / Holes", use_container_width=True)
        b2.image(dilated_bridge.astype(np.uint8) * 255, caption="Gaps Bridged / Holes Filled", use_container_width=True)
        
        st.success("Dilation effectively 'repairs' fragmented objects by expanding their boundaries until they touch.")

    with tab_gray:
        st.subheader("Grayscale Dilation (Max Filter)")
        st.markdown("Bright regions expand, dark valleys get filled.")
        
        # Load a grayscale version
        if local_up:
            gray_img = np.array(Image.open(io.BytesIO(local_up.getvalue())).convert('L'))
        else:
            # Synthetic grayscale for demo
            gray_img = np.zeros((400, 400), dtype=np.uint8)
            gray_img[100:300, 100:300] = 100
            # Bright spots in dark region
            gray_img[50, 50] = 200
            gray_img[350, 350] = 200
            
        gray_dilated = morphology.dilation(gray_img, se)
        
        g1, g2 = st.columns(2)
        g1.image(gray_img, caption="Original Grayscale", use_container_width=True)
        g2.image(gray_dilated, caption="Grayscale Dilated (Maximum Filter)", use_container_width=True)
        
        st.caption("Grayscale dilation expands bright features and 'erases' small dark noise.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Dilation** ($\oplus$): The 'Grower' of morphology.
    - **Binary**: $B$ must touch at least one pixel of $A$.
    - **Results**: Thickens objects, fills internal holes, bridges narrow gaps.
    - **Duality**: Dual of Erosion ($A \oplus B = (A^c \ominus \hat{B})^c$).
    - **Grayscale**: Equivalent to a nonlinear 'Maximum Filter'.
    """)
