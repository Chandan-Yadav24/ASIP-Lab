# demos/haar_features_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, feature, color, data, util, exposure
from PIL import Image
import io

@st.cache_data
def load_img_haar(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Use a face image for relevant Haar feature demonstration
        img_np = data.astronaut()
        img_np = color.rgb2gray(img_np)
        # Crop to the face area roughly
        img_np = img_np[30:180, 150:300]
    return img_np

def compute_integral_image(img):
    # Integral image: II(x, y) = sum of intensities above and to the left
    return np.cumsum(np.cumsum(img, axis=0), axis=1)

def run():
    st.header("🎭 Haar-like Features (Viola–Jones)")
    st.markdown("""
    Haar-like features are simple rectangular stencils used to detect local patterns of light and dark. They are the engine behind real-time face detection.
    """)

    with st.expander("📚 Theory: The Rectangle Sum Trick", expanded=False):
        st.markdown(r"""
        ### 1. The Core Idea
        A Haar-like feature subtracts the sum of pixels in a dark rectangle from the sum in a light rectangle:
        $$Value = \sum(\text{White Region}) - \sum(\text{Black Region})$$
        
        ### 2. The Integral Image (Summed-Area Table)
        To compute these sums instantly, we use an **Integral Image**. 
        - Every pixel $(x, y)$ in the integral image stores the sum of all pixels above and to its left.
        - **The Trick**: Any rectangular sum can be found using just 4 lookups:
          $$Sum = D + A - (B + C)$$
        
        ### 3. Feature Types
        - **Edge Features**: Detect horizontal/vertical transitions.
        - **Line Features**: Detect thin structures (like the nose bridge).
        - **Four-rectangle**: Detect diagonal/checkerboard changes.
        
        ### 🎭 The Stencil Analogy
        Imagine sliding simple black-and-white cardboard stencils over a photo. If the light parts of the skin align with the white parts of the stencil, the "score" is high. Thousands of these simple scores combined can identify a face!
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="haar_local")
    img = load_img_haar(local_up.read() if local_up else None)
    img_gray = util.img_as_float(img)

    tab_integral, tab_stencil, tab_table = st.tabs(["📶 Integral Image", "📐 Interactive Stencils", "📊 Feature Types"])

    with tab_integral:
        st.subheader("The Integral Image Visualizer")
        st.markdown("This 'Summed-Area Table' allows for lightning-fast rectangle calculations.")
        
        ii = compute_integral_image(img_gray)
        
        col1, col2 = st.columns(2)
        col1.image(img_gray, caption="Original Intensity", use_container_width=True)
        # Rescale II for visualization
        ii_vis = exposure.rescale_intensity(ii, out_range=(0, 1))
        col2.image(ii_vis, caption="Integral Image (Cumulative Sums)", use_container_width=True)
        
        st.info("Notice how the integral image gets brighter towards the bottom-right corner as the cumulative sum grows.")

    with tab_stencil:
        st.subheader("Manual Feature Probe")
        st.markdown("Slide a stencil over the image and see the response.")
        
        f_type = st.selectbox("Choose Stencil Pattern", ["Vertical Edge (2-rect)", "Horizontal Edge (2-rect)", "Line Feature (3-rect)"])
        
        # Interactive coordinates
        h, w = img_gray.shape
        y = st.slider("Vertical Position", 0, h-40, h//2)
        x = st.slider("Horizontal Position", 0, w-40, w//2)
        size = st.slider("Stencil Size", 10, 40, 20)
        
        fig, ax = plt.subplots()
        ax.imshow(img_gray, cmap='gray')
        
        # Calculate response (Simplified logic for visualization)
        if f_type == "Vertical Edge (2-rect)":
            # Left Black, Right White
            rect_b = plt.Rectangle((x, y), size//2, size, color='black', alpha=0.5)
            rect_w = plt.Rectangle((x + size//2, y), size//2, size, color='white', alpha=0.5)
            val = np.sum(img_gray[y:y+size, x+size//2:x+size]) - np.sum(img_gray[y:y+size, x:x+size//2])
        elif f_type == "Horizontal Edge (2-rect)":
            # Top Black, Bottom White
            rect_b = plt.Rectangle((x, y), size, size//2, color='black', alpha=0.5)
            rect_w = plt.Rectangle((x, y + size//2), size, size//2, color='white', alpha=0.5)
            val = np.sum(img_gray[y+size//2:y+size, x:x+size]) - np.sum(img_gray[y:y+size//2, x:x+size])
        else: # 3-rect line
            # Outer Black, Inner White
            rect_b = plt.Rectangle((x, y), size, size, color='black', alpha=0.5)
            rect_w = plt.Rectangle((x + size//3, y), size//3, size, color='white', alpha=0.5)
            val = np.sum(img_gray[y:y+size, x+size//3:x+x+2*size//3]) - (np.sum(img_gray[y:y+size, x:x+size//3]) + np.sum(img_gray[y:y+size, x+2*size//3:x+size]))

        ax.add_patch(rect_b)
        ax.add_patch(rect_w)
        ax.axis('off')
        st.pyplot(fig)
        
        st.metric("Feature Response Value", round(val, 2))
        st.caption("A high positive value means the image matches the stencil pattern well at this location.")

    with tab_table:
        st.subheader("Common Haar Stencil Types")
        st.table({
            "Feature Type": ["Edge Features", "Line Features", "3-Rectangle Features", "4-Rectangle Features"],
            "Pattern Layout": ["Horizontal/Vertical split", "Bright/Dark line", "Center vs Surrounds", "Diagonal differences"],
            "Target Object": ["Eyes vs Eyebrows", "Nose bridge", "Horizontal textures", "Diagonal structures"]
        })
        
        st.markdown("""
        ### Why Use Haar?
        Even though modern Deep Learning (CNNs) is more accurate, **Haar Cascades** remain useful for:
        - Battery-powered devices (IoT)
        - Low-latency real-time video tracking
        - Simple embedded systems
        """)

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Haar Feature**: Difference between rectangular area sums.
    - **Integral Image**: The speed boost! Constant time sums regardless of size.
    - **Cascade**: A series of classifiers where the easy non-objects are rejected instantly.
    - **Invariance**: Robust to intensity shifts (due to differences) but sensitive to rotation/scale.
    """)
