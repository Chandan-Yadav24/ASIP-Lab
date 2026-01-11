# demos/fft_mechanics_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fft2, fftshift
from PIL import Image
import io

@st.cache_data
def get_processed_image(file_data, size=(64, 64)):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img = np.array(img.resize(size)) / 255.0
    else:
        x = np.linspace(0, 5, size[0])
        y = np.linspace(0, 5, size[1])
        X, Y = np.meshgrid(x, y)
        img = (np.sin(X) + np.cos(Y) + 2) / 4
    return img

@st.cache_data
def compute_separability_fft(img):
    row_fft = np.fft.fft(img, axis=1) 
    full_2d = np.fft.fft(row_fft, axis=0)
    return row_fft, full_2d

def run():
    st.subheader("FFT Mechanics & Computational Efficiency")

    # --- Utility: Safe Normalization -----------------------------------------
    def norm_img(data):
        d_min, d_max = np.min(data), np.max(data)
        if d_max > d_min:
            return np.clip((data - d_min) / (d_max - d_min), 0, 1)
        return np.clip(data, 0, 1)

    # --- Theory Section ------------------------------------------------------
    with st.expander("📚 Theory: Why FFT is a Miracle", expanded=False):
        st.markdown(r"""
        The **Fast Fourier Transform (FFT)** isn't a different transform—it's just a **very clever way** to compute the Discrete Fourier Transform (DFT).
        
        ### 1) The Complexity Gap
        - **Brute-force DFT**: $O(N \cdot N)$. Every sample is compared with every frequency base.
        - **FFT**: $O(N \cdot \log N)$. Uses **Divide and Conquer** to skip redundant math.
        
        ### 2) Successive Doubling (Mechanics)
        The FFT splits a signal into **Even** and **Odd** samples, computes their transforms separately, and then recombines them using "Butterfly" operations. This recursive structure is what creates the speedup.
        
        ### 3) 2D Separability
        For an $M \times N$ image, we don't need a single global 2D transform. We can:
        1. Apply 1D FFT to all **Rows**.
        2. Apply 1D FFT to all **Columns** of the previous result.
        """)

    st.markdown("---")

    tab_eff, tab_sep, tab_audio, tab_analogy = st.tabs(["🚀 Efficiency Explorer", "🧩 Separability Lab", "🎵 Audio FFT", "📦 Sorting Analogy"])

    # --- Global Input Layer (Shared for Separability) -----------------------
    st.sidebar.write("### 📥 Image Input")
    uploaded_file = st.sidebar.file_uploader("Upload Image for FFT", type=['jpg', 'jpeg', 'png'], key="fft_mech_upload")
    
    file_bytes = uploaded_file.read() if uploaded_file else None
    img_mini = get_processed_image(file_bytes)

    # --- Tab 1: Efficiency Explorer ------------------------------------------
    with tab_eff:
        st.write("### The Complexity Gap: $N^2$ vs $N \log N$")
        
        n_val = st.select_slider("Select Image Side N (Total pixels = N*N)", 
                                 options=[64, 128, 256, 512, 1024, 2048, 4096], value=512)
        
        total_p = n_val**2
        dft_ops = total_p**2
        fft_ops = total_p * np.log2(total_p) if total_p > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Pixels", f"{total_p:,}")
        m2.metric("Brute-force Ops", f"{dft_ops:.1e}")
        m3.metric("FFT Ops", f"{fft_ops:.1e}")
        
        st.info(f"For a **{n_val}x{n_val}** image, the FFT is approximately **{int(dft_ops/fft_ops):,}x faster** than a direct calculation.")
        
        # Plotting the gap
        ns = np.array([32, 64, 128, 256, 512, 1024])
        pixels = ns**2
        fig_c, ax_c = plt.subplots(figsize=(8, 4))
        ax_c.plot(pixels, pixels**2, 'o--', label="Brute DFT", color='tab:red')
        ax_c.plot(pixels, pixels * np.log2(pixels), 'o-', label="FFT", color='tab:green')
        ax_c.set_yscale('log')
        ax_c.set_xlabel("Number of Pixels")
        ax_c.set_ylabel("Operation Count (Log Scale)")
        ax_c.legend()
        ax_c.grid(alpha=0.3)
        st.pyplot(fig_c)

    # --- Tab 2: Separability Lab ---------------------------------------------
    with tab_sep:
        st.write("### 2D FFT: Step-by-Step Separability")
        st.markdown(f"Using **{64}x{64}** downsampled version of your input for visibility.")
        
        stage = st.radio("FFT Progress Stage", ["0. Original Image", "1. Vertical FFT (Across Rows)", "2. Vertical + Horizontal (Final 2D FFT)"], key="sep_stage")
        
        # Row-Column decomposition
        row_fft, full_2d = compute_separability_fft(img_mini)
        
        c_v1, c_v2 = st.columns([1, 1.5])
        with c_v1:
            if stage[0] == "0":
                disp = img_mini
                title = "Spatial Domain"
            elif stage[0] == "1":
                disp = norm_img(np.log(1 + np.abs(fftshift(row_fft))))
                title = "Row FFTs (Hybrid)"
            else:
                disp = norm_img(np.log(1 + np.abs(fftshift(full_2d))))
                title = "2D Spectrum"
            
            st.image(disp, caption=title, use_container_width=True)

        with c_v2:
            st.markdown(f"**Insight for Stage {stage[0]}:**")
            if stage[0] == "0":
                st.write("Input pixel grid.")
            elif stage[0] == "1":
                st.write("Each row is now a 1D Fourier Transform. We see frequency content horizontally, but still vertical spatial structure.")
            else:
                st.write("Now columns are also transformed. The entire grid is in the frequency domain.")
            
            error = np.sum(np.abs(full_2d - fft2(img_mini)))
            st.success(f"Equivalence Check: Error = {error:.1e}")

    # --- Tab 3: Audio FFT ----------------------------------------------------
    with tab_audio:
        st.write("### Audio FFT: The Sound of Frequencies")
        st.markdown("Transforming sound into a spectrum and back.")
        
        freq_a = st.slider("Signal Frequency (Hz)", 200, 2000, 440)
        duration = 1.0
        fs = 8000
        t_a = np.linspace(0, duration, fs, endpoint=False)
        sig_a = 0.5 * np.sin(2 * np.pi * freq_a * t_a)
        
        # Noise toggle
        if st.checkbox("Add High-Frequency Noise"):
            sig_a += 0.2 * np.sin(2 * np.pi * 3500 * t_a)
        
        f_sig = fft(sig_a)
        f_abs = np.abs(f_sig[:fs//2])
        f_freqs = np.linspace(0, fs//2, fs//2)
        
        fig_a, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 6))
        ax_t.plot(t_a[:200], sig_a[:200])
        ax_t.set_title("Time Domain (First 200 samples)")
        ax_f.plot(f_freqs, f_abs)
        ax_f.set_title("Frequency Domain (FFT Magnitude)")
        ax_f.set_xlim(0, 4000)
        plt.tight_layout()
        st.pyplot(fig_a)
        
        # Audio playback
        st.audio(sig_a, format="audio/wav", sample_rate=fs)
        st.caption("Listen to the synthesized signal.")

    # --- Tab 4: Analogy ------------------------------------------------------
    with tab_analogy:
        st.write("### 📚 The Sorting Library Analogy")
        st.markdown("""
        **Sorting 1 Million Books:**
        - **Brute Force**: Comparing every book with every other book.
        - **FFT Strategy**: 
            1. Divide books into Even and Odd shelves.
            2. Further divide into smaller piles.
            3. Sort tiny piles.
            4. Merge sorted results back up.
        """)
        st.latex(r"X[k] = E[k] + e^{-j\frac{2\pi}{N}k} O[k]")
        st.caption("The recursive 'Butterfly' merge operation.")

    # --- Summary Table -------------------------------------------------------
    st.markdown("---")
    st.write("### Complexity Summary")
    st.table({
        "Method": ["Brute DFT", "FFT"],
        "Ops (1K Image)": ["1.1 Trillion", "21 Million"],
        "Speed Gain": ["1x", "~50,000x"],
        "Implementation": ["Separable Row-Col", "Successive Doubling"]
    })
