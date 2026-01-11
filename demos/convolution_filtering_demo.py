# demos/convolution_filtering_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image
import io

@st.cache_data
def get_base_data(file_data):
    if file_data:
        img_raw = Image.open(io.BytesIO(file_data)).convert('L')
        img_base = np.array(img_raw.resize((256, 256))) / 255.0
    else:
        x_d = np.linspace(0, 1, 256)
        y_d = np.linspace(0, 1, 256)
        X_d, Y_d = np.meshgrid(x_d, y_d)
        img_base = 0.5 + 0.5 * np.sin(2 * np.pi * X_d) * np.sin(2 * np.pi * Y_d)
    return img_base

@st.cache_data
def get_base_fft(img_base):
    return fftshift(fft2(img_base))

def run():
    st.subheader("Convolution & Spectral Filtering (Advanced)")

    # --- Utility: Safe Normalization -----------------------------------------
    def norm_img(data):
        d_min, d_max = np.min(data), np.max(data)
        if d_max > d_min:
            return np.clip((data - d_min) / (d_max - d_min), 0, 1)
        return np.clip(data, 0, 1)

    # --- Theory Section ------------------------------------------------------
    with st.expander("📚 Theory: Spectral Multiplication", expanded=False):
        st.markdown(r"""
        **Convolution Theorem**: $f(x, y) * h(x, y) \iff F(u, v) \cdot H(u, v)$.
        
        ### Key Concepts
        1. **Ideal Filter**: Sharp cutoff. Causes **Gibbs ringing** (periodic artifacts).
        2. **Butterworth Filter**: Smooth transition. Order $n$ controls the "steepness".
        3. **Gaussian Filter**: No ringing. Purely smooth smoothing/sharpening.
        4. **Notch Filters**: Surgeons for the spectrum. Used to remove specific periodic frequencies.
        """)

    st.markdown("---")

    # --- Global Input Layer --------------------------------------------------
    st.write("### 📥 Global Input Layer")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="conv_global_upload")
    
    file_bytes = uploaded_file.read() if uploaded_file else None
    img_base = get_base_data(file_bytes)
    F_base = get_base_fft(img_base)

    tab1, tab2, tab3 = st.tabs(["1. Filter Profiles", "2. Periodic Noise Lab", "3. Spectral Sharpening"])

    # Pre-compute distance grid for filters
    rows, cols = img_base.shape
    crow, ccol = rows//2, cols//2
    y_idx, x_idx = np.ogrid[:rows, :cols]
    dist_map = np.sqrt((x_idx - ccol)**2 + (y_idx - crow)**2)

    # --- Tab 1: Filter Profiles ----------------------------------------------
    with tab1:
        st.write("### Spectral Mask Designer")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            f_type = st.selectbox("Design Type", ["Ideal", "Butterworth", "Gaussian"], key="p_type")
            p_mode = st.radio("Behavior", ["Low-pass (Blur)", "High-pass (Detail)"], key="p_mode")
            d0 = st.slider("Cutoff Radius", 5, 120, 40, key="p_d0")
            
            # Transfer Function Calculation
            if f_type == "Ideal":
                H = (dist_map <= d0).astype(float)
            elif f_type == "Butterworth":
                n = st.slider("Butterworth Order", 1, 5, 2, key="p_order")
                H = 1 / (1 + (dist_map / d0)**(2*n))
            else: # Gaussian
                H = np.exp(-(dist_map**2) / (2 * (d0**2)))
                
            if p_mode == "High-pass (Detail)":
                H = 1 - H

        with c2:
            G = F_base * H
            res = np.abs(ifft2(fftshift(G)))
            
            fig_p, (ax_h, ax_res) = plt.subplots(1, 2, figsize=(10, 4))
            ax_h.imshow(H, cmap='magma')
            ax_h.set_title("Filter Mask $H(u, v)$")
            ax_h.axis('off')
            
            ax_res.imshow(res, cmap='gray')
            ax_res.set_title(f"Resulting {p_mode}")
            ax_res.axis('off')
            st.pyplot(fig_p)

    # --- Tab 2: Periodic Noise Lab -------------------------------------------
    with tab2:
        st.write("### Periodic Noise Removal (Notch Control)")
        
        # Part A: Corrupt the image
        st.markdown("**1. Inject Noise**")
        n_col1, n_col2 = st.columns(2)
        with n_col1:
            noise_f = st.slider("Noise Frequency", 5, 100, 30, key="noise_f")
            noise_a = st.slider("Noise Amplitude", 0.1, 1.0, 0.4, key="noise_a")
            # Sine noise
            noise_img = noise_a * np.sin(2 * np.pi * noise_f * (x_idx + y_idx) / 256.0)
            img_corrupt = np.clip(img_base + noise_img, 0, 1)
            st.image(img_corrupt, caption="Corrupted Signal", use_container_width=True)

        with n_col2:
            F_noise = fftshift(fft2(img_corrupt))
            spec_view = np.log(1 + np.abs(F_noise))
            st.image(norm_img(spec_view), caption="Spectrum (Notice the Spikes)", use_container_width=True)

        # Part B: Notch Filter
        st.markdown("**2. Tune Notch Filter**")
        st.info("Move the notches to 'hide' the bright spikes in the spectrum.")
        
        notch_radius = st.slider("Notch Radius", 1, 20, 5, key="nr")
        
        # Manual Notch Positioning
        # For our generated noise, we know where it is, but let's give the user control
        u_notch = st.slider("Notch U (X Offset)", -100, 100, noise_f, key="u_n")
        v_notch = st.slider("Notch V (Y Offset)", -100, 100, noise_f, key="v_n")
        
        # Create Notch Mask
        H_notch = np.ones_like(img_base)
        # Periodic noise has symmetric spikes (+/- freq)
        dist1 = np.sqrt((x_idx - (ccol + u_notch))**2 + (y_idx - (crow + v_notch))**2)
        dist2 = np.sqrt((x_idx - (ccol - u_notch))**2 + (y_idx - (crow - v_notch))**2)
        
        H_notch[dist1 <= notch_radius] = 0
        H_notch[dist2 <= notch_radius] = 0
        
        F_clean = F_noise * H_notch
        img_clean = np.abs(ifft2(fftshift(F_clean)))
        
        res1, res2 = st.columns(2)
        res1.image(norm_img(np.log(1 + np.abs(F_clean))), caption="Cleaned Spectrum", use_container_width=True)
        res2.image(norm_img(img_clean), caption="Restored Image", use_container_width=True)

    # --- Tab 3: Sharpening ---------------------------------------------------
    with tab3:
        st.write("### High-Boost Filtering")
        st.markdown("Sharpening by boosting high-frequency components.")
        
        k = st.slider("Sharpening Strength", 0.0, 3.0, 1.0, key="sharpen_k")
        # Sharpening Mask: 1 + k * HPF
        # Using Gaussian HPF for smoothness
        H_lp = np.exp(-(dist_map**2) / (2 * (30**2)))
        H_hp = 1 - H_lp
        H_boost = 1 + k * H_hp
        
        res_boost = np.abs(ifft2(fftshift(F_base * H_boost)))
        st.image(norm_img(res_boost), caption=f"Sharpened Result (Strength: {k})", use_container_width=True)

    # --- Analogy: Colored Sands ----------------------------------------------
    st.markdown("---")
    st.write("### 🏖️ The Colored Sands Analogy")
    st.markdown("""
    - **Smoothing**: Removing the coarse, tiny grains (high frequencies) to leave only the soft sand.
    - **Notching**: Surgically picking out a specific group of colored grains (periodic noise) that shouldn't be there.
    - **Sharpening**: Sifting out everything *but* the edges, then mixing them back in to make the textures pop.
    """)

