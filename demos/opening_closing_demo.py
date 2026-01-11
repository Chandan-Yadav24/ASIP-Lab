# demos/opening_closing_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology
import io

@st.cache_data
def load_img_morph(file_data, threshold=128, add_noise=False, add_gaps=False):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
        binary = img_np > threshold
    else:
        # Synthetic base: a few large circles and rectangles
        img_np = np.zeros((400, 400), dtype=bool)
        # Rect
        img_np[100:300, 50:150] = True
        # Circle
        rr, cc = np.ogrid[:400, :400]
        mask = (rr - 200)**2 + (cc - 300)**2 <= 70**2
        img_np[mask] = True
        binary = img_np

    if add_noise:
        # Salt noise (white specks for Opening test)
        noise = np.random.rand(*binary.shape) > 0.99
        binary = binary | noise
    
    if add_gaps:
        # Crack/Gap (for Closing test)
        binary[190:210, 50:350] = False
        
    return binary

def get_se(shape, size):
    if shape == "Square":
        return morphology.square(size)
    elif shape == "Disk":
        return morphology.disk(size // 2 if size > 1 else 1)
    else: # Cross
        return morphology.star(size // 2 if size > 1 else 1)

def run():
    st.header("🧽 Opening & Closing: Smart Cleaning")
    st.markdown("""
    Opening and Closing are **compound operations** that combine erosion and dilation to clean images without drastically changing the size of large objects.
    """)

    with st.expander("📚 Theory: Clean & Repair", expanded=False):
        st.markdown(r"""
        ### 1. Opening ($A \circ B$)
        **Formula**: $(A \ominus B) \oplus B$ (Erosion then Dilation)
        - **Intuition**: "Brush off the dust."
        - **Effect**: Removes small bright noise and thin protrusions. 
        - **Key**: Things completely removed by erosion don't come back with dilation.
        
        ### 2. Closing ($A \bullet B$)
        **Formula**: $(A \oplus B) \ominus B$ (Dilation then Erosion)
        - **Intuition**: "Fill the cracks."
        - **Effect**: Fills small dark holes and bridges narrow gaps.
        - **Key**: Gaps filled by dilation remain filled after erosion.
        
        ### 3. Duality
        - Opening foreground is like closing the background.
        - They are dual operations with respect to complementation.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="morph_local")
    
    # Mode selection for synthetic image
    test_mode = st.sidebar.radio("Synthetic Test Case", ["Clean", "Salt Noise (for Opening)", "Broken Gaps (for Closing)"])
    
    add_n = test_mode == "Salt Noise (for Opening)"
    add_g = test_mode == "Broken Gaps (for Closing)"
    
    thresh_val = 128
    if local_up:
        thresh_val = st.sidebar.slider("Binary Threshold", 0, 255, 128)
        
    binary_img = load_img_morph(local_up.read() if local_up else None, threshold=thresh_val, add_noise=add_n, add_gaps=add_g)
    
    st.sidebar.markdown("### 🧰 Structuring Element (SE)")
    se_shape = st.sidebar.selectbox("Shape", ["Square", "Disk", "Cross"])
    se_size = st.sidebar.slider("Size", 1, 15, 5, step=2)
    se = get_se(se_shape, se_size)

    tab_open, tab_close, tab_step = st.tabs(["🧹 Opening Lab", "🩹 Closing Lab", "🎞️ Step-by-Step"])

    with tab_open:
        st.subheader("Opening: Noise Removal")
        st.markdown("Removes small objects but keeps large ones at their original size.")
        
        opened = morphology.binary_opening(binary_img, se)
        
        o1, o2 = st.columns(2)
        o1.image(binary_img.astype(np.uint8) * 255, caption="Original / Noisy", use_container_width=True)
        o2.image(opened.astype(np.uint8) * 255, caption="After Opening", use_container_width=True)
        
        st.info("The small white specks disappear, but the large shapes stay roughly the same size.")

    with tab_close:
        st.subheader("Closing: Gap Repair")
        st.markdown("Fills small holes and bridges narrow cracks/gaps.")
        
        closed = morphology.binary_closing(binary_img, se)
        
        c1, c2 = st.columns(2)
        c1.image(binary_img.astype(np.uint8) * 255, caption="Original / Broken", use_container_width=True)
        c2.image(closed.astype(np.uint8) * 255, caption="After Closing", use_container_width=True)
        
        st.success("Dilation filled the gap, and erosion brought the object back to its correct boundary.")

    with tab_step:
        st.subheader("Mechanics: The Two-Step Process")
        op_type = st.radio("Visualize:", ["Opening (Erosion ➜ Dilation)", "Closing (Dilation ➜ Erosion)"])
        
        if "Opening" in op_type:
            step1 = morphology.binary_erosion(binary_img, se)
            step2 = morphology.binary_dilation(step1, se)
            captions = ["1. Original", "2. Erosion (Shrunk/Cleared)", "3. Dilation (Restored Size)"]
        else:
            step1 = morphology.binary_dilation(binary_img, se)
            step2 = morphology.binary_erosion(step1, se)
            captions = ["1. Original", "2. Dilation (Grown/Filled)", "3. Erosion (Restored Size)"]
            
        s1, s2, s3 = st.columns(3)
        s1.image(binary_img.astype(np.uint8) * 255, caption=captions[0], use_container_width=True)
        s2.image(step1.astype(np.uint8) * 255, caption=captions[1], use_container_width=True)
        s3.image(step2.astype(np.uint8) * 255, caption=captions[2], use_container_width=True)
        
        st.markdown(f"**Notice**: The final result is much closer to the original size than the intermediate step 2.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Opening** ($\circ$): Erosion $\to$ Dilation. Cleans small bright objects.
    - **Closing** ($\bullet$): Dilation $\to$ Erosion. Fills small dark holes/gaps.
    - **Smoothing**: Both operations smooth contours (Opening rounds convex corners, Closing rounds concave ones).
    - **Idempotency**: Applying opening/closing a second time on the result has no further effect.
    """)
