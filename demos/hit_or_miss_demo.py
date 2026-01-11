# demos/hit_or_miss_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
import io

@st.cache_data
def load_img_hmt(file_data, threshold=128):
    if file_data:
        from PIL import Image
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
        binary = img_np > threshold
    else:
        # Synthetic binary image with various corners and lines
        img_np = np.zeros((200, 200), dtype=bool)
        # Large Square
        img_np[50:150, 50:150] = True
        # Small rectangle
        img_np[30:40, 30:100] = True
        # Isolated pixel
        img_np[10, 10] = True
        # Line
        img_np[160:190, 20:21] = True
        binary = img_np
    return binary

def hit_or_miss(image, hit_se, miss_se):
    """Perform Hit-or-Miss transformation."""
    # Hit: Foreground fits
    hit_part = binary_erosion(image, structure=hit_se)
    # Miss: Background fits (erosion of complement)
    image_c = ~image
    miss_part = binary_erosion(image_c, structure=miss_se)
    # Result: Intersection
    return hit_part & miss_part

def run():
    st.header("🎯 Hit-or-Miss Transformation: Pattern Detector")
    st.markdown("""
    The **Hit-or-Miss Transformation (HMT)** is used to find very specific pixel configurations. 
    It checks both the foreground (Hit) and the background (Miss) at the same time.
    """)

    with st.expander("📚 Theory: The Logic of Exact Matches", expanded=False):
        st.markdown(r"""
        ### 1. The Core Idea
        Erosion alone only checks the foreground. HMT adds a second condition: the background around the shape must also match a specific pattern.
        
        ### 2. Formula
        $$HMT(A, B) = (A \ominus C) \cap (A^c \ominus D)$$
        - $A$: Input Binary Image.
        - $C$: **Hit** Structuring Element (Foreground pattern).
        - $D$: **Miss** Structuring Element (Background pattern).
        - $\cap$: Intersection (Both must be True).
        
        ### 3. Usage
        - Locate corners, endpoints, or junctions.
        - Detect specific shapes (e.g., a "T" junction or a isolated pixel).
        - Preprocessing for thinning or skeletonization.
        """)

    tab_toy, tab_search, tab_step = st.tabs(["🔢 Toy Grid Lab", "🔍 Feature Search", "🎞️ Logic Steps"])

    with tab_toy:
        st.subheader("3x3 Numerical Example")
        st.markdown("Toggle the pixels in the input grid to see where the **Vertical Line** pattern is detected.")
        
        # Grid input using columns and buttons
        if 'grid' not in st.session_state:
            st.session_state.grid = np.array([
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 0]
            ], dtype=bool)
            
        cols = st.columns(3)
        for r in range(3):
            for c in range(3):
                with cols[c]:
                    if st.button(f"{'⬛' if st.session_state.grid[r,c] else '⬜'}", key=f"btn_{r}_{c}"):
                        st.session_state.grid[r,c] = not st.session_state.grid[r,c]
                        st.rerun()

        # Vertical Line SEs
        # Hit: center and top must be 1
        hit_se = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=bool)
        # Miss: sides and bottom must be 0
        miss_se = np.array([
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=bool)
        
        result = hit_or_miss(st.session_state.grid, hit_se, miss_se)
        
        c1, c2 = st.columns(2)
        c1.markdown("**Current Image $A$**")
        c1.write(st.session_state.grid.astype(int))
        c2.markdown("**HMT Output (Match detected at '1')**")
        c2.write(result.astype(int))
        
        st.info("The HMT outputs 1 only at the location where the *entire* 3x3 neighborhood matches the pattern.")

    with tab_search:
        st.subheader("Shape Detection Lab")
        st.sidebar.markdown("### 📥 Image Lab")
        local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="hmt_local")
        binary_img = load_img_hmt(local_up.read() if local_up else None)
        
        pattern = st.selectbox("Find Pattern:", ["Upper-Left Corner", "Bottom Endpoint", "Isolated Pixel"])
        
        if pattern == "Upper-Left Corner":
            hit_se = np.array([[0,0,0],[0,1,1],[0,1,1]], bool)
            miss_se = np.array([[1,1,1],[1,0,0],[1,0,0]], bool)
        elif pattern == "Bottom Endpoint":
            hit_se = np.array([[0,1,0],[0,1,0],[0,0,0]], bool)
            miss_se = np.array([[1,0,1],[1,0,1],[1,1,1]], bool)
        else: # Isolated Pixel
            hit_se = np.array([[0,0,0],[0,1,0],[0,0,0]], bool)
            miss_se = np.array([[1,1,1],[1,0,1],[1,1,1]], bool)
            
        detected = hit_or_miss(binary_img, hit_se, miss_se)
        
        # Dilate result for display visibility
        from scipy.ndimage import binary_dilation
        detected_disp = binary_dilation(detected, structure=np.ones((5,5)))
        
        col1, col2 = st.columns(2)
        col1.image(binary_img.astype(np.uint8)*255, caption="Input Binary", use_container_width=True)
        # Highlight matches on original
        overlay = binary_img.astype(float) * 0.5
        overlay[detected_disp] = 1.0
        col2.image(overlay, caption="Detected Locations (Highlighted)", use_container_width=True)

    with tab_step:
        st.subheader("Visualizing $(A \ominus C) \cap (A^c \ominus D)$")
        
        hit_e = binary_erosion(binary_img, structure=hit_se)
        image_c = ~binary_img
        miss_e = binary_erosion(image_c, structure=miss_se)
        
        s1, s2, s3 = st.columns(3)
        s1.image(hit_e.astype(np.uint8)*255, caption="1. Hit: Foreground fits (A ⊖ C)", use_container_width=True)
        s2.image(miss_e.astype(np.uint8)*255, caption="2. Miss: Background fits (A^c ⊖ D)", use_container_width=True)
        s3.image(detected.astype(np.uint8)*255, caption="3. Result: Logic AND (1 ∩ 2)", use_container_width=True)

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Hit-or-Miss Transformation**:
    - **Logic**: Exact pattern matching of Foreground (Hit) AND Background (Miss).
    - **Formula**: $A \otimes B = (A \ominus C) \cap (A^c \ominus D)$.
    - **Elements**: Composite SE $B = (C, D)$ where $C \cap D = \emptyset$.
    - **Usage**: Find corners, junctions, endpoints, or specific pixel clusters.
    """)
