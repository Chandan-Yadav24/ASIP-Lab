# demos/boundary_extraction_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, color, util
from PIL import Image
import io

@st.cache_data
def load_img_boundary(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
        binary = img_np > 128
    else:
        # Synthetic binary shapes
        img_np = np.zeros((300, 300), dtype=bool)
        # Rectangle
        img_np[50:150, 50:150] = True
        # Circle
        yy, xx = np.mgrid[:300, :300]
        mask = (yy - 200)**2 + (xx - 200)**2 <= 60**2
        img_np[mask] = True
        # Hollow Triangle
        for i in range(50):
            img_np[50+i, 200-i:200+i] = True
        binary = img_np
    return binary

def run():
    st.header("➰ Boundary Extraction: The Morphological Outline")
    st.markdown("""
    Boundary extraction is a technique used to find the **contour** or outline of an object by subtracting its "eroded core" from the original shape.
    """)

    with st.expander("📚 Theory: The Hollow Shape", expanded=False):
        st.markdown(r"""
        ### 1. The Core Formula
        $$\text{Boundary}(A) = A - (A \ominus B)$$
        - **$A$**: Original binary image.
        - **$B$**: Structuring Element (SE).
        - **$A \ominus B$**: Erosion of A by B (the shrunk version).
        
        ### 2. Inner vs Outer Boundary
        - **Inner Boundary** (shown here): $A - (A \ominus B)$. The outline lies **inside** the original object's footprint.
        - **Outer Boundary**: $(A \oplus B) - A$. The outline lies **outside** the original footprint (requires dilation).
        
        ### 3. Thickness Control
        The size of the structuring element $B$ determines the thickness of the boundary.
        - A 3x3 square SE results in a **1-pixel thick** inner boundary.
        - A larger SE creates a broader, more "hollowed" look.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="bound_local")
    binary_img = load_img_boundary(local_up.read() if local_up else None)

    st.sidebar.markdown("### 🧰 Structuring Element")
    se_size = st.sidebar.slider("Thickness (SE Size)", 1, 15, 3, step=2)
    se = morphology.square(se_size)

    tab_lab, tab_steps, tab_comparison = st.tabs(["➰ Extraction Lab", "🎞️ Step-by-Step", "🆚 Inner vs Outer"])

    # Perform Inner Boundary
    eroded = morphology.binary_erosion(binary_img, se)
    boundary_inner = binary_img ^ eroded # XOR or subtraction

    with tab_lab:
        st.subheader("Interactive Outline Generator")
        st.markdown(f"Current Structuring Element: Square ({se_size}x{se_size})")
        
        col1, col2 = st.columns(2)
        col1.image(binary_img.astype(np.uint8)*255, caption="Original Binary", use_container_width=True)
        col2.image(boundary_inner.astype(np.uint8)*255, caption=f"Extracted Boundary ({se_size}px thickness)", use_container_width=True)
        
        st.info("By subtracting the 'shrunk' version, only the boundary pixels that were erased remain.")

    with tab_steps:
        st.subheader("Logic Visualization")
        
        s1, s2, s3 = st.columns(3)
        s1.image(binary_img.astype(np.uint8)*255, caption="1. Original (A)", use_container_width=True)
        s2.image(eroded.astype(np.uint8)*255, caption="2. Eroded (A ⊖ B)", use_container_width=True)
        s3.image(boundary_inner.astype(np.uint8)*255, caption="3. Result (A - (A ⊖ B))", use_container_width=True)
        
        st.markdown("**Observation**: The result in Step 3 is literally the 'difference' between the solid object and its core.")

    with tab_comparison:
        st.subheader("Inner Boundary vs Outer Boundary")
        
        # Outer Boundary = Dilation - Original
        dilated = morphology.binary_dilation(binary_img, se)
        boundary_outer = dilated ^ binary_img
        
        c1, c2 = st.columns(2)
        c1.image(boundary_inner.astype(np.uint8)*255, caption="Inner Boundary (Inside Object)", use_container_width=True)
        c2.image(boundary_outer.astype(np.uint8)*255, caption="Outer Boundary (Outside Object)", use_container_width=True)
        
        st.caption("Inner boundary subtracts pixels from the inside edge. Outer boundary adds pixels to the outside edge.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Boundary Extraction**:
    - **Formula**: $A \setminus (A \ominus B)$.
    - **Method**: Subtract eroded version from original.
    - **Result**: A one-pixel-thick contour (if SE is 3x3).
    - **Usage**: Calculating perimeters, finding edges, shape analysis.
    - **Parameter**: SE size directly controls the border thickness.
    """)
