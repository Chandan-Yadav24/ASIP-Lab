# demos/dft_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fft2, ifft2, fftshift
from PIL import Image
import io

@st.cache_data
def load_and_resize_img(file_data, size=(256, 256)):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img.resize(size))
    else:
        x = np.linspace(0, 5, size[0])
        y = np.linspace(0, 5, size[1])
        X, Y = np.meshgrid(x, y)
        img_np = (np.sin(X*Y) * 127 + 128).astype(np.uint8)
    return img_np

@st.cache_data
def compute_fft2_centered(img_np):
    return fftshift(fft2(img_np))

def run():
    st.subheader("Discrete Fourier Transform (DFT): Analyzing the Ingredients")

    # --- Theory Section (Infographic Style) ----------------------------------
    with st.expander("📚 Theory: The DFT Transform Pair", expanded=False):
        st.markdown(r"""
        The **DFT** converts a signal into its constituent sines and cosines.
        
        ### 1) The Math
        - **Forward DFT**: $F(u) = \sum_{x=0}^{M-1} f(x) e^{-j2\pi ux/M}$
        - **Inverse DFT**: Recover $f(x)$ from $F(u)$. It is a reversible process!
        
        ### 2) Key Properties
        *   **Periodicity**: The DFT assumes the finite signal repeats forever.
        *   **Separability (2D)**: A 2D DFT is just a series of 1D DFTs (Rows, then Columns).
        *   **Complexity**: The **FFT** (Fast Fourier Transform) makes this computation incredibly efficient.
        
        ### 3) Frequency Meaning
        *   **Low Freq**: Base shapes, average light, smooth areas.
        *   **High Freq**: Edges, fine texture, sharp changes, noise.
        """)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["1D Signal DFT", "2D Image Filtering", "FFT Compression Lab"])

    # --- Tab 1: 1D DFT --------------------------------------------------------
    with tab1:
        st.write("### 1D DFT: Magnitude and Phase")
        col_1d1, col_1d2 = st.columns([1, 2])
        
        with col_1d1:
            f1 = st.slider("Frequency 1 (Hz)", 1, 50, 10, key="dft_f1")
            f2 = st.slider("Frequency 2 (Hz)", 50, 150, 80, key="dft_f2")
            n_pts = 512
            t = np.linspace(0, 1, n_pts, endpoint=False)
            sig = np.sin(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t)
            
        with col_1d2:
            # Compute DFT
            X = fft(sig)
            freqs = np.fft.fftfreq(n_pts) * n_pts # Sample frequencies
            
            fig_1d, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 6))
            ax_t.plot(t[:100], sig[:100], color='tab:blue')
            ax_t.set_title("Input Signal (First 100 samples)")
            ax_t.grid(alpha=0.2)
            
            # Show positive frequencies only
            pos_mask = freqs >= 0
            ax_f.stem(freqs[pos_mask][:150], np.abs(X)[pos_mask][:150], basefmt=" ")
            ax_f.set_title("DFT Magnitude Spectrum")
            ax_f.set_xlabel("Frequency Bin / Hz")
            ax_f.grid(alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_1d)

    # --- Tab 2: 2D Image Filtering --------------------------------------------
    with tab2:
        st.write("### 2D DFT Lab: Frequency-Domain Filters")
        st.info("Workflow: Image $\to$ FFT $\to$ Multiply by Mask $\to$ IFFT")
        
        img_file = st.file_uploader("Upload image for 2D DFT", type=['jpg', 'png'], key="dft_img_upload")
        
        file_bytes = img_file.read() if img_file else None
        img_np = load_and_resize_img(file_bytes)

        f_type = st.radio("Filter Mode", ["None", "Low-Pass (Blur)", "High-Pass (Edges)"], key="dft_f_mode")
        radius = st.slider("Filter Radius", 5, 100, 30, key="dft_radius")

        # Process
        F = compute_fft2_centered(img_np)
        rows, cols = img_np.shape
        crow, ccol = rows//2, cols//2
        
        # Create mask
        y_grid, x_grid = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((x_grid - ccol)**2 + (y_grid - crow)**2)
        
        if f_type == "Low-Pass (Blur)":
            mask = dist_from_center <= radius
        elif f_type == "High-Pass (Edges)":
            mask = dist_from_center > radius
        else:
            mask = np.ones_like(F)

        # Helper to safely normalize for st.image
        def norm_img(data):
            d_max = np.max(data)
            if d_max > 0:
                return np.clip(data / d_max, 0, 1)
            return np.clip(data, 0, 1)

        F_filtered = F * mask
        img_back = np.abs(ifft2(fftshift(F_filtered)))

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.image(img_np, caption="Input (Spatial)", use_container_width=True)
        with col_p2:
            st.image(norm_img(np.log(1 + np.abs(F_filtered))), caption="Spectrum (Filtered)", use_container_width=True)
        with col_p3:
            st.image(norm_img(img_back), caption="Result (Post-Inverse)", use_container_width=True)

    # --- Tab 3: Compression ---------------------------------------------------
    with tab3:
        st.write("### Image Compression: The Sparsity Power")
        st.markdown("We can discard small frequency coefficients to compress data.")
        
        keep_percent = st.slider("Coefficients to Keep (%)", 0.1, 100.0, 5.0, key="dft_compress")
        
        # 2D FFT
        F_comp = fft2(img_np)
        F_flat = np.abs(F_comp.flatten())
        
        # Find threshold
        threshold = np.percentile(F_flat, 100 - keep_percent)
        F_thresholded = F_comp * (np.abs(F_comp) >= threshold)
        
        img_compressed = np.abs(ifft2(F_thresholded))
        
        cc1, cc2 = st.columns(2)
        cc1.image(img_np, caption="Original (100% Data)", use_container_width=True)
        cc2.image(norm_img(img_compressed), caption=f"Compressed ({keep_percent}% Data)", use_container_width=True)
        
        st.success(f"Compression complete! Using only **{keep_percent}%** of frequency components.")

    # --- Analogy: Smoothie ---------------------------------------------------
    st.markdown("---")
    st.write("### 🥤 The Smoothie Analogy")
    st.markdown("""
    - **Original Signal**: The blended smoothie (everything mixed).
    - **Forward DFT**: The lab analysis that finds the % of strawberry, banana, and sugar.
    - **Filtering**: Removing the "sugar" (noise) from the ingredient list.
    - **Inverse DFT**: Blending the modified ingredient list back into a healthier smoothie.
    """)
