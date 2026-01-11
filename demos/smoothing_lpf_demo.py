# demos/smoothing_lpf_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image

def run():
    st.subheader("Smoothing with Low-Pass Filters (LPF)")

    # --- Utility: Safe Normalization -----------------------------------------
    def norm_img(data):
        d_min, d_max = np.min(data), np.max(data)
        if d_max > d_min:
            return np.clip((data - d_min) / (d_max - d_min), 0, 1)
        return np.clip(data, 0, 1)

    # --- Theory Section ------------------------------------------------------
    with st.expander("📚 Theory: The Sieve of Frequencies", expanded=False):
        st.markdown(r"""
        **Low-Pass Filters (LPF)** act as frequency sieves that keep slow transitions (backgrounds) while discarding rapid jumps (edges/noise).
        
        ### 1) The Main Contenders
        *   **Ideal LPF**: Perfectly sharp. *Problem*: Causes **Gibbs Ringing** (ripples near edges).
        *   **Gaussian LPF**: Mathematically smooth. *Benefit*: No ringing; naturally preferred for photos.
        *   **Butterworth LPF**: Flexible. Toggle order $n$ to trade-off sharpness and ringing.
        
        ### 2) The Relationship
        Higher **Cutoff Frequency $D_0$** $\implies$ More detail preserved, less blur.
        Lower **Cutoff Frequency $D_0$** $\implies$ More noise removed, more blur.
        """)

    st.markdown("---")

    # --- Global Input Layer --------------------------------------------------
    st.write("### 📥 Image & Noise Layer")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="smooth_upload")
    
    if uploaded_file:
        img_raw = Image.open(uploaded_file).convert('L')
        img_base = np.array(img_raw.resize((256, 256))) / 255.0
    else:
        # Default text image (good for showing blur/ringing)
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        X, Y = np.meshgrid(x, y)
        img_base = np.zeros((256, 256))
        img_base[100:156, 50:200] = 1.0 # White block on black
        
    noise_lvl = st.slider("Add Noise (Gaussian)", 0.0, 0.5, 0.1, key="smooth_noise")
    img_noisy = np.clip(img_base + np.random.normal(0, noise_lvl, img_base.shape), 0, 1)

    tab1, tab2, tab3 = st.tabs(["1. The Ringing Test", "2. Noise Mitigation", "3. Gentle Rain (3D)"])

    # Pre-compute distance grid
    rows, cols = img_base.shape
    crow, ccol = rows//2, cols//2
    idx_y, idx_x = np.ogrid[:rows, :cols]
    dist = np.sqrt((idx_x - ccol)**2 + (idx_y - crow)**2)

    # --- Tab 1: Ringing Test --------------------------------------------------
    with tab1:
        st.write("### The Gibbs Phenomenon: Ringing vs. Smoothness")
        st.info("Ideal filters cause 'ghost' ripples. Gaussian filters do not.")
        
        d0_ring = st.slider("Cutoff Radius", 10, 120, 30, key="ring_d0")
        
        # ILPF
        H_ideal = (dist <= d0_ring).astype(float)
        # GLPF
        H_gauss = np.exp(-(dist**2) / (2 * (d0_ring**2)))
        
        # Apply to non-noisy base (shows ringing better)
        F = fftshift(fft2(img_base))
        res_ideal = np.abs(ifft2(fftshift(F * H_ideal)))
        res_gauss = np.abs(ifft2(fftshift(F * H_gauss)))
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(norm_img(res_ideal), caption="Ideal LPF (Notice Ripples)", use_container_width=True)
            st.caption("Sharp edges cause the Fourier components to 'bounce'.")
        with c2:
            st.image(norm_img(res_gauss), caption="Gaussian LPF (Smooth Blur)", use_container_width=True)
            st.caption("Smooth transition in frequency domain = Smooth results in space.")

    # --- Tab 2: Noise Mitigation ---------------------------------------------
    with tab2:
        st.write("### Noise Reduction Lab")
        st.markdown("Filter out the high-frequency 'grain' of the noisy image.")
        
        f_type = st.radio("Sieve Type", ["Gaussian LPF", "Butterworth LPF"], horizontal=True)
        d0_noise = st.select_slider("Select Pass-radius", options=np.arange(5, 120, 5), value=20)
        
        F_noisy = fftshift(fft2(img_noisy))
        
        if f_type == "Gaussian LPF":
            H_n = np.exp(-(dist**2) / (2 * (d0_noise**2)))
        else:
            order = st.slider("Filter Order", 1, 10, 2)
            H_n = 1 / (1 + (dist / d0_noise)**(2 * order))
            
        res_n = np.abs(ifft2(fftshift(F_noisy * H_n)))
        
        cn1, cn2 = st.columns(2)
        cn1.image(norm_img(img_noisy), caption="Noisy Input", use_container_width=True)
        cn2.image(norm_img(res_n), caption="Filtered Output", use_container_width=True)

    # --- Tab 3: Gentle Rain --------------------------------------------------
    with tab3:
        st.write("### Analogy: The Landscape under Rain")
        st.markdown("High-frequency noise = Jagged Peaks. LPF = Gentle Erosion.")

        # Downsample for 3D performance
        z_small = img_noisy[::4, ::4]
        z_smooth = res_n[::4, ::4]
        h_s, w_s = z_small.shape
        x_s, y_s = np.meshgrid(np.arange(w_s), np.arange(h_s))
        
        fig_3d = plt.figure(figsize=(10, 8))
        ax1 = fig_3d.add_subplot(121, projection='3d')
        ax2 = fig_3d.add_subplot(122, projection='3d')
        
        ax1.plot_surface(x_s, y_s, z_small, cmap='viridis', edgecolor='none')
        ax1.set_title("Jagged Terrain (Noisy)")
        ax1.axis('off')
        
        ax2.plot_surface(x_s, y_s, z_smooth, cmap='viridis', edgecolor='none')
        ax2.set_title("Eroded Terrain (Smoothed)")
        ax2.axis('off')
        
        st.pyplot(fig_3d)

    # --- Summary Table -------------------------------------------------------
    st.markdown("---")
    st.write("### Summary of LPF Types")
    st.table({
        "Filter": ["Ideal (ILPF)", "Gaussian (GLPF)", "Butterworth (BLPF)"],
        "Advantage": ["Sharpest cutoff", "Zero Ringing (Stable)", "Flexible Trade-off"],
        "Artifacts": ["High Ringing", "Broad Blur", "N-dependent Ringing"],
        "Best Use": ["Math analysis", "Medical/Photography", "General Purpose"]
    })
