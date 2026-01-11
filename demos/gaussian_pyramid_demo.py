# demos/gaussian_pyramid_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
import io

@st.cache_data
def load_img_pyramid(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic image with varying detail scales
        img_np = np.zeros((512, 512), dtype=np.uint8)
        # Checkerboard (fine detail)
        for i in range(0, 512, 32):
            for j in range(0, 512, 32):
                if (i//32 + j//32) % 2 == 0:
                    img_np[i:i+32, j:j+32] = 200
        # Large circle (coarse structure)
        y, x = np.ogrid[:512, :512]
        mask = (x - 256)**2 + (y - 256)**2 <= 180**2
        img_np[mask] = img_np[mask] // 2 + 50
        img = Image.fromarray(img_np)
    return np.array(img)

def build_gaussian_pyramid(img, levels, sigma):
    """Build a Gaussian pyramid with specified levels."""
    pyramid = [img.astype(float)]
    current = img.astype(float)
    for _ in range(levels - 1):
        # Blur
        blurred = gaussian_filter(current, sigma=sigma)
        # Subsample (every 2nd pixel)
        subsampled = blurred[::2, ::2]
        pyramid.append(subsampled)
        current = subsampled
    return pyramid

def run():
    st.header("🏔️ Gaussian Pyramid: Multi-Scale Image Representation")
    st.markdown("""
    A **Gaussian Pyramid** represents an image at multiple resolutions by repeatedly **blurring** and **subsampling**.
    """)

    with st.expander("📚 Theory: How It Works", expanded=False):
        st.markdown(r"""
        ### 1. Construction
        Starting from original image $G_0$:
        1. **Blur**: $G_i^{blur} = G_i * G_\sigma$
        2. **Subsample**: Keep every 2nd pixel → $G_{i+1}$
        3. Repeat for each level.

        ### 2. Why Blur Before Subsampling?
        - Prevents **aliasing** (high-freq → fake low-freq patterns).
        - Creates a clean, meaningful low-res version.

        ### 3. Properties
        - **Efficient**: Each level has ~1/4 the pixels of the previous.
        - **Multi-Scale**: Fine details at bottom, coarse structure at top.
        - **Foundation**: Basis for Laplacian Pyramid and scale-space analysis.

        ### 4. Applications
        - **Image Blending**: Smooth transitions at multiple scales.
        - **Coarse-to-Fine Search**: Fast matching at low-res, refine at high-res.
        - **Feature Detection**: SIFT, texture analysis.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="pyr_local")
    img_gray = load_img_pyramid(local_up.read() if local_up else None)

    tab_build, tab_compare, tab_alias = st.tabs(["🏗️ Build Pyramid", "🔍 Level Comparison", "⚠️ Aliasing Demo"])

    with tab_build:
        st.subheader("Interactive Pyramid Builder")
        
        levels = st.slider("Number of Levels", 2, 6, 4)
        sigma = st.slider("Blur Sigma", 0.5, 3.0, 1.0)
        
        pyramid = build_gaussian_pyramid(img_gray, levels, sigma)
        
        # Display as a visual pyramid
        fig, axes = plt.subplots(1, levels, figsize=(12, 3))
        for i, level_img in enumerate(pyramid):
            axes[i].imshow(level_img, cmap='gray', vmin=0, vmax=255)
            axes[i].set_title(f"Level {i}\n{level_img.shape[0]}×{level_img.shape[1]}")
            axes[i].axis('off')
        
        st.pyplot(fig)
        st.info(f"Total pixels: Level 0 = {pyramid[0].size:,}, Level {levels-1} = {pyramid[-1].size:,} (Compression: {pyramid[0].size / pyramid[-1].size:.1f}x)")

    with tab_compare:
        st.subheader("Fine vs Coarse Structure")
        
        pyramid = build_gaussian_pyramid(img_gray, 5, 1.0)
        
        c1, c2 = st.columns(2)
        c1.image(pyramid[0].astype(np.uint8), caption="Level 0 (Full Detail)", use_container_width=True)
        c2.image(pyramid[-1].astype(np.uint8), caption=f"Level {len(pyramid)-1} (Coarse Only)", use_container_width=True)
        
        st.markdown("""
        - **Level 0**: All fine details visible (checkerboard pattern).
        - **Top Level**: Only large structures remain (the circle).
        """)

    with tab_alias:
        st.subheader("Why Blur? Aliasing Prevention")
        
        # Without blur (direct subsample)
        direct_sub = img_gray[::8, ::8]
        
        # With blur (proper pyramid level)
        blurred = gaussian_filter(img_gray, sigma=3.0)
        proper_sub = blurred[::8, ::8]
        
        a1, a2 = st.columns(2)
        a1.image(direct_sub.astype(np.uint8), caption="Direct Subsample (No Blur)", use_container_width=True)
        a1.caption("❌ Artifacts: Fake patterns from aliasing")
        
        a2.image(proper_sub.astype(np.uint8), caption="Blur + Subsample (Correct)", use_container_width=True)
        a2.caption("✅ Clean: High-freq removed before downsampling")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Gaussian Pyramid**: Multi-resolution via repeated Blur + Subsample.
    - **Structure**: $G_0 \\to G_1 \\to G_2 \\to ...$
    - **Key**: Gaussian blur prevents aliasing.
    - **Uses**: Blending, coarse-to-fine search, basis for Laplacian pyramid.
    """)
