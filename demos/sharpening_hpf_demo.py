# demos/sharpening_hpf_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import laplace
from PIL import Image

def run():
    st.subheader("Sharpening with High-Pass Filters (HPF)")

    # --- Utility: Safe Normalization -----------------------------------------
    def norm_img(data):
        d_min, d_max = np.min(data), np.max(data)
        if d_max > d_min:
            return np.clip((data - d_min) / (d_max - d_min), 0, 1)
        return np.clip(data, 0, 1)

    # --- Theory Section ------------------------------------------------------
    with st.expander("📚 Theory: Extracting the Edges", expanded=False):
        st.markdown(r"""
        **High-Pass Filters (HPF)** are the inverse of Low-Pass filters. They discard the coarse background (DC component) and keep the fine, rapid changes (Edges).
        
        ### 1) The Relationship
        $H_{HP}(u, v) = 1 - H_{LP}(u, v)$
        
        ### 2) Core Sharpening Techniques
        *   **Laplacian**: A second-derivative operator. It highlights discontinuities.
        *   **Unsharp Masking**: $f_{sharp} = f_{orig} + k(f_{orig} - f_{blur})$. We subtract the blur (leaving only detail) and add it back.
        *   **High-Boost Filtering**: Using $k > 1$ for aggressive detail enhancement.
        *   **High-Frequency Emphasis**: Adding a constant offset to the HPF to preserve overall image brightness.
        """)

    st.markdown("---")

    # --- Global Image Layer --------------------------------------------------
    st.write("### 📥 Input Selection")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="hpf_upload")
    
    if uploaded_file:
        img_raw = Image.open(uploaded_file).convert('L')
        img_base = np.array(img_raw.resize((256, 256))) / 255.0
    else:
        # Default architectural image concept (high details)
        x = np.linspace(0, 5, 256)
        y = np.linspace(0, 5, 256)
        X, Y = np.meshgrid(x, y)
        img_base = norm_img(np.cos(X**2 + Y**2))
        
    st.image(img_base, caption="Original Image", width=300)

    tab1, tab2, tab3 = st.tabs(["1. HPF Gallery", "2. Unsharp Mask & High-Boost", "3. Laplacian Sharpening"])

    # Pre-compute distance grid
    rows, cols = img_base.shape
    crow, ccol = rows//2, cols//2
    idx_y, idx_x = np.ogrid[:rows, :cols]
    dist = np.sqrt((idx_x - ccol)**2 + (idx_y - crow)**2)

    # --- Tab 1: HPF Gallery --------------------------------------------------
    with tab1:
        st.write("### The Detail Extractors")
        st.info("High-pass filters remove the 'meat' of the image, leaving only the 'bones' (edges).")
        
        h_type = st.selectbox("HPF Transfer Function", ["Ideal", "Butterworth", "Gaussian"], key="gal_type")
        cutoff = st.slider("Cutoff Frequency", 5, 100, 30, key="gal_d0")
        
        # Calculate HPF
        if h_type == "Ideal":
            H_lp = (dist <= cutoff).astype(float)
        elif h_type == "Butterworth":
            n = st.slider("Filter Order", 1, 5, 2)
            H_lp = 1 / (1 + (dist / cutoff)**(2*n))
        else: # Gaussian
            H_lp = np.exp(-(dist**2) / (2 * (cutoff**2)))
            
        H_hp = 1 - H_lp
        
        # Apply FFT
        F = fftshift(fft2(img_base))
        res_hp = np.abs(ifft2(fftshift(F * H_hp)))
        
        # High-Frequency Emphasis (Optional Toggle)
        if st.checkbox("Apply High-Frequency Emphasis (+ DC Offset)"):
            H_hf = 0.5 + 0.75 * H_hp
            res_hp = np.abs(ifft2(fftshift(F * H_hf)))

        gh1, gh2 = st.columns(2)
        with gh1:
            fig_h, ax_h = plt.subplots()
            ax_h.imshow(H_hp, cmap='magma')
            ax_h.axis('off')
            ax_h.set_title(f"{h_type} HPF Mask")
            st.pyplot(fig_h)
        
        gh2.image(norm_img(res_hp), caption="High Frequency Content", use_container_width=True)

    # --- Tab 2: Unsharp Masking ----------------------------------------------
    with tab2:
        st.write("### Unsharp Masking & High-Boost")
        st.markdown(r"Algorithm: $f_{sharp} = f_{orig} + k \cdot (f_{orig} - f_{blur})$")
        
        k = st.slider("Sharpening Strength", 0.0, 5.0, 1.0, key="usk_k")
        blur_rad = st.slider("Blur Radius (Smoothing)", 1, 15, 3, key="usk_r")
        
        # Create blur (for manual UM logic)
        H_blur = np.exp(-(dist**2) / (2 * (blur_rad**2)))
        img_blur = np.abs(ifft2(fftshift(F * H_blur)))
        
        # Create mask
        mask = img_base - img_blur
        img_sharp = img_base + k * mask
        
        # Visualization
        cu1, cu2, cu3 = st.columns(3)
        cu1.image(norm_img(img_blur), caption="Step 1: Blurred", use_container_width=True)
        cu2.image(norm_img(mask + 0.5), caption="Step 2: Detail Mask", use_container_width=True)
        cu3.image(norm_img(img_sharp), caption="Step 3: Sharpened Result", use_container_width=True)
        
        if k > 1.0:
            st.warning(f"High-Boost mode active ($k={k}$). Notice how edges become much more aggressive.")

    # --- Tab 3: Laplacian Lab ------------------------------------------------
    with tab3:
        st.write("### Laplacian Edge Enhancement")
        st.markdown("The Laplacian highlights points of rapid second-order changes.")
        
        # Compute Laplacian
        res_lap = laplace(img_base)
        
        # Enhancement: Original - Res (using standard convention for central pixel)
        # Or Original + Res depending on kernel sign. We'll use a subtractive boost.
        boost = st.slider("Laplacian Boost", 0.1, 2.0, 1.0)
        img_enhanced = img_base - boost * res_lap # Standard sharpening
        
        cl1, cl2 = st.columns(2)
        cl1.image(norm_img(res_lap + 0.5), caption="Pure Laplacian output", use_container_width=True)
        cl2.image(norm_img(img_enhanced), caption="Enhanced with Laplacian", use_container_width=True)
        
    # --- Analogy: Pencil Drawing ---------------------------------------------
    st.markdown("---")
    st.write("### ✏️ The Pencil Drawing Analogy")
    st.markdown("""
    - **Original Signal**: A faint, blurry pencil sketch where everything looks gray and soft.
    - **High-Pass Filtering**: Using a sharp black pencil to **trace** only the outlines and fine details.
    - **Sharpening**: Overlaying that sharp tracing back onto the original sketch so the important lines "pop" without losing the overall form.
    """)
