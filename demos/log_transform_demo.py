# demos/log_transform_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, fftshift
import io

@st.cache_data
def load_img_log(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A very dark image with hidden structure
        x = np.linspace(0, 10, 512)
        y = np.linspace(0, 10, 512)
        X, Y = np.meshgrid(x, y)
        # Low intensity signal
        img_np = (5 * (np.sin(X) + np.cos(Y) + 2)).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)

@st.cache_data
def apply_log_transform(img, c):
    # s = c * log(1 + r)
    # We work with normalized r [0, 1] for better scaling control
    r = img.astype(float) / 255.0
    s = c * np.log(1 + r)
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)

def run():
    st.header("📉 Log Transformation: The Dark Detail Expander")
    st.markdown("""
    The log transform is a powerful tool for visual enhancement. It stretches the dark parts of an image while squeezing the bright parts.
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: The Math of Log Curves", expanded=False):
        st.markdown(r"""
        ### 1. Mathematical Definition
        $s = c \log(1 + r)$
        - $r$: Input pixel intensity (assumed $\geq 0$).
        - $s$: Output pixel intensity.
        - $c$: Scaling constant.

        **Why $(1+r)$?**  
        In digital images, a pixel can be exactly $0$. Since $\log(0)$ is undefined ($-\infty$), we add $1$ to ensure the math always works.

        ### 2. Visual Impact
        1. **Expansion of Darks**: A small range of low intensities is stretched into a large output range.
        2. **Compression of Brights**: A large range of high intensities is squeezed into a small output range.

        ### 3. The Sunglasses Analogy 🕶️
        Imagine a sunset so bright you can't see the trees (silhouettes). 
        - The log transform acts like **Smart Sunglasses**.
        - It dims the glare (brights) while "opening up" the details in the shadows (darks).
        """)

    # --- Global Input ---
    st.sidebar.markdown("### 📥 Image Source")
    local_up = st.file_uploader("Upload Image (Optional)", type=['png', 'jpg', 'jpeg'], key="log_local")
    img_gray = load_img_log(local_up.read() if local_up else None)
    
    h_orig, w_orig = img_gray.shape
    st.info(f"📁 **Active Image**: {w_orig}x{h_orig} | Base range: [{np.min(img_gray)}, {np.max(img_gray)}]")

    tab_lab, tab_spectrum, tab_lut = st.tabs(["🧪 Transformation Lab", "📡 Fourier Spectrum Lab", "📊 Curve & LUT"])

    with tab_lab:
        st.subheader("Interactive scaling (Constant c)")
        c_val = st.slider("Scaling Constant (c)", 0.1, 5.0, 1.0, step=0.1)
        
        res_log = apply_log_transform(img_gray, c_val)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_gray, caption="Original Image (Often very dark)", use_container_width=True)
            st.caption("Input range is small/compressed.")
        with col2:
            st.image(res_log, caption=f"Log Transformed (c={c_val})", use_container_width=True)
            st.caption("Details in shadows are now visible!")

    with tab_spectrum:
        st.subheader("Key Application: Fourier Spectra")
        st.markdown("""
        The magnitude of a Fourier Transform often has values ranging from $0$ to $10^{10}$. 
        Without a Log Transform, you'd only see a single white dot in the middle!
        """)
        
        # Compute FFT
        F = fftshift(fft2(img_gray))
        mag = np.abs(F)
        
        # Raw view (usually looks like one dot)
        mag_norm = mag / (np.max(mag) if np.max(mag) > 0 else 1)
        
        # Log view
        mag_log = np.log(1 + mag)
        mag_log_norm = mag_log / (np.max(mag_log) if np.max(mag_log) > 0 else 1)
        
        c_spec1, c_spec2 = st.columns(2)
        with c_spec1:
            st.image(mag_norm, caption="Raw Magnitude Spectrum", use_container_width=True)
            st.error("Too much dynamic range! Only the DC component (center) is visible.")
        with c_spec2:
            st.image(mag_log_norm, caption="Log-Compressed Spectrum", use_container_width=True)
            st.success("Log transform 'flattened' the range so we see the ripples!")

    with tab_lut:
        st.subheader("LUT (Look-Up Table) Logic")
        st.markdown("""
        Because $\log$ is expensive to calculate for millions of pixels, we precompute a **256-entry table**.
        """)
        
        r_range = np.linspace(0, 1, 256)
        s_curve = c_val * np.log(1 + r_range)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(r_range * 255, np.clip(s_curve, 0, 1) * 255, color='orange', lw=3, label="Log Mapping")
        ax.plot([0, 255], [0, 255], 'k--', alpha=0.3, label="Linear (Identity)")
        
        ax.set_title(f"Intensity Mapping Curve (c={c_val})")
        ax.set_xlabel("Input Intensity (r)")
        ax.set_ylabel("Output Intensity (s)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.info("The curvature above represents the 'Sunglasses' effect—stretching darks (low r) rapidly into middle shades (s).")

    # Final Summary
    st.divider()
    st.markdown("### 📋 Log Transform Profile")
    st.table({
        "Feature": ["Formula", "Dark Pixels", "Bright Pixels", "Best For", "Complexity"],
        "Details": ["s = c log(1+r)", "Expanded (visible)", "Compressed (dimmed)", "Spectrum & Medical Lab", "O(1) with LUT"]
    })
