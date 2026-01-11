# demos/grayscale_morphology_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, color, util
from PIL import Image
import io

@st.cache_data
def load_img_gray_morph(file_data, test_case="Synthetic"):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        if test_case == "Shapes & Peaks":
            # Grayscale background with bright peaks and dark valleys
            img_np = np.full((300, 300), 128, dtype=np.uint8)
            yy, xx = np.mgrid[:300, :300]
            # Smooth gradient
            img_np = (img_np + 40 * np.sin(xx/40)).astype(np.uint8)
            # Bright shapes
            img_np[50:120, 50:120] = 220
            # Dark shapes
            img_np[180:250, 180:250] = 30
            # Small noise specks
            for _ in range(20):
                img_np[np.random.randint(0, 298, 2).tolist()] = 255
                img_np[np.random.randint(0, 298, 2).tolist()] = 0
        else: # Text
             # Create simple text
             img = Image.new('L', (300, 150), 200)
             from PIL import ImageDraw
             draw = ImageDraw.Draw(img)
             draw.text((20, 20), "MORPH", fill=50) # Dark text on light
             img_np = np.array(img)
    return img_np

def run():
    st.header("🌫️ Grayscale Morphology: Intensity Operations")
    st.markdown("""
    Grayscale morphology extends binary operations to intensity images by using **Min/Max** operations over a local neighborhood defined by a Structuring Element (SE).
    """)

    with st.expander("📚 Theory: Min, Max, and Gradients", expanded=False):
        st.markdown(r"""
        ### 1. Fundamental Operations
        - **Erosion (Min Filter)**: $ (f \ominus b)(x,y) = \min_{(s,t) \in b} \{ f(x+s, y+t) \} $
          - Bright objects shrink, dark regions expand.
        - **Dilation (Max Filter)**: $ (f \oplus b)(x,y) = \max_{(s,t) \in b} \{ f(x-s, y-t) \} $
          - Bright objects grow, dark regions shrink.
        
        ### 2. Sequence Operations
        - **Opening**: Erosion followed by Dilation. Removes small bright details.
        - **Closing**: Dilation followed by Erosion. Fills small dark details.
        
        ### 3. Edge Detection
        - **Morphological Gradient**: $ (f \oplus b) - (f \ominus b) $
          - The difference between Dilation and Erosion highlights sharp transitions (edges).
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    test_case = st.sidebar.selectbox("Test Case", ["Shapes & Peaks", "Text"])
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="gray_morph_local")
    
    img = load_img_gray_morph(local_up.read() if local_up else None, test_case=test_case)

    st.sidebar.markdown("### 🧰 Structuring Element")
    se_shape = st.sidebar.selectbox("SE Shape", ["Disk", "Square", "Diamond"])
    se_size = st.sidebar.slider("Radius / Size", 1, 15, 3)
    
    if se_shape == "Disk":
        se = morphology.disk(se_size)
    elif se_shape == "Square":
        se = morphology.square(se_size * 2 + 1)
    else:
        se = morphology.diamond(se_size)

    tab_basics, tab_clean, tab_grad = st.tabs(["📉 Min/Max Filter", "🧹 Opening & Closing", "⚡ Edge Gradient"])

    with tab_basics:
        st.subheader("Grayscale Erosion vs Dilation")
        
        eroded = morphology.erosion(img, se)
        dilated = morphology.dilation(img, se)
        
        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="Original Grayscale", use_container_width=True)
        c2.image(eroded, caption="Eroded (Min Filter)", use_container_width=True)
        c3.image(dilated, caption="Dilated (Max Filter)", use_container_width=True)
        
        st.info("Erosion darkens the image by picking the minimum neighbor. Dilation brightens it by picking the maximum.")

    with tab_clean:
        st.subheader("Intensity Smoothing")
        
        opened = morphology.opening(img, se)
        closed = morphology.closing(img, se)
        
        s1, s2, s3 = st.columns(3)
        s1.image(img, caption="Original", use_container_width=True)
        s2.image(opened, caption="Opened (Bright detail removed)", use_container_width=True)
        s3.image(closed, caption="Closed (Dark detail filled)", use_container_width=True)
        
        st.success("Opening is great for removing bright 'salt' noise. Closing is great for filling dark 'pepper' noise.")

    with tab_grad:
        st.subheader("Morphological Gradient")
        st.markdown(r"Gradient = $\text{Dilation} - \text{Erosion}$")
        
        # Calculate gradient manually to show the logic
        e = morphology.erosion(img, se).astype(float)
        d = morphology.dilation(img, se).astype(float)
        grad = d - e
        
        g1, g2 = st.columns(2)
        g1.image(img, caption="Original Image", use_container_width=True)
        g2.image(grad.astype(np.uint8), caption="Morphological Gradient (Edges)", use_container_width=True)
        
        st.warning("The gradient highlights sharp intensity changes, serving as a non-linear edge detector.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Grayscale Morphology**:
    - **Erosion**: Neighborhood Minimum. Darkens bright spikes.
    - **Dilation**: Neighborhood Maximum. Brightens dark valleys.
    - **Opening**: $f \circ b$. Suppresses small bright details.
    - **Closing**: $f \bullet b$. Suppresses small dark details.
    - **Gradient**: $f \oplus b - f \ominus b$. High-pass filter for edge detection.
    """)
