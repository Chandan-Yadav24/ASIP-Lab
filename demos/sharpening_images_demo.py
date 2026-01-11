# demos/sharpening_images_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace, convolve
from scipy.fft import fft2, ifft2, fftshift
import io

@st.cache_data
def load_img_sharpen(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A slightly blurred shapes image
        img_np = np.zeros((512, 512), dtype=np.uint8)
        img_np[150:350, 150:350] = 180
        # Add a softer edge blur initially
        img_np = gaussian_filter(img_np, sigma=2.0)
        img = Image.fromarray(img_np)
    return np.array(img)

def frequency_hpf(img, cutoff, order=2, type="Butterworth"):
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    if type == "Ideal":
        mask = (dist > cutoff).astype(float)
    elif type == "Gaussian":
        mask = 1 - np.exp(-(dist**2) / (2 * (cutoff**2)))
    else: # Butterworth
        mask = 1 / (1 + (cutoff / (dist + 1e-10))**(2 * order))
        
    f_coef = fftshift(fft2(img))
    f_filtered = f_coef * mask
    img_back = np.abs(ifft2(fftshift(f_filtered)))
    return np.clip(img_back, 0, 255).astype(np.uint8)

def run():
    st.header("✏️ Sharpening: The Pencil Trace")
    st.markdown("""
    Sharpening highlights edges and detail. It emphasizes discontinuities by calculating derivatives—mathematically, it's the opposite of smoothing.
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: Differentiation & Details", expanded=False):
        st.markdown(r"""
        ### 1. Spatial Domain (Derivatives)
        - **1st Order (Gradients)**: Responds to the *rate* of change (Roberts, Sobel).
        - **2nd Order (Laplacian)**: Highlight discontinuities regardless of direction. 
          Equation: $g = f + c[\nabla^2 f]$
        
        ### 2. Unsharp Masking & High-Boost
        A classic 3-step process:
        1. **Blur** the original image.
        2. Create a **Mask** (Original - Blur).
        3. **Add** the mask back to the original: $g = f + k(f - f_{blur})$
           - $k=1$: Unsharp Masking.
           - $k > 1$: High-Boost Filtering (Reveals even more detail).

        ### 3. The Pencil Trace Analogy ✏️
        If smoothing is like rubbing a sketch with a soft cloth, sharpening is like using a fine dark pencil to **trace** existing lines. It doesn't add new data; it just makes boundaries stand out!

        ### 4. Frequency Domain (HPF)
        High-Pass Filters (HPF) let through the "fast" changes (edges) and block the "slow" ones (backgrounds).
        """)

    # --- Global Input ---
    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="sharpen_local")
    
    img_gray = load_img_sharpen(local_up.read() if local_up else None)
    st.info(f"📁 **Active Image**: {'Uploaded' if local_up else 'Blurred Synthetic'}")

    tab_spatial, tab_mask, tab_freq = st.tabs(["🧪 Spatial Laplacian Lab", "🎭 Unsharp & High-Boost", "🌊 Frequency HPF Lab"])

    with tab_spatial:
        st.subheader("The Laplacian Operator")
        st.markdown("Highlights sharp intensity transitions.")
        
        c_weight = st.slider("Sharpening Weight (c)", -2.0, 2.0, -1.0, step=0.1)
        
        # Laplacian calculation
        lap = laplace(img_gray.astype(float))
        # The equation: g = f - nabla^2 f (if the center of laplacian kernel is negative)
        # scipy.ndimage.laplace uses [0, 1, 0, 1, -4, 1, 0, 1, 0] approx
        img_sharp_sp = np.clip(img_gray.astype(float) + c_weight * lap, 0, 255).astype(np.uint8)
        
        col1, col2 = st.columns(2)
        col1.image(img_gray, caption="Original", use_container_width=True)
        col2.image(img_sharp_sp, caption=f"Laplacian Sharpened (c={c_weight})", use_container_width=True)
        
        st.caption("Lower c (more negative) usually strengthens edges for standard 3x3 kernels.")

    with tab_mask:
        st.subheader("Unsharp Masking & High-Boost")
        
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            sigma = st.slider("Blur Intensity (Sigma)", 0.5, 5.0, 1.5)
            k_val = st.slider("Boost Factor (k)", 0.0, 5.0, 1.0)
            
            # Step 1: Blur
            blur = gaussian_filter(img_gray.astype(float), sigma=sigma)
            # Step 2: Mask
            mask = img_gray.astype(float) - blur
            # Step 3: Add back
            img_hb = np.clip(img_gray.astype(float) + k_val * mask, 0, 255).astype(np.uint8)
            
            if k_val == 1.0:
                st.success("Target: Standard Unsharp Masking")
            elif k_val > 1.0:
                st.warning("Target: High-Boost Filtering")
            
        with col_m2:
            st.image(img_hb, caption=f"Result (k={k_val})", use_container_width=True)
        
        st.divider()
        st.image(np.clip(mask + 128, 0, 255).astype(np.uint8), caption="The Sharpening Mask (Level shifted for visibility)", use_container_width=True)

    with tab_freq:
        st.subheader("Frequency Domain High-Pass")
        st.markdown("Directly filtering the Fourier Spectrum.")
        
        f_cutoff = st.slider("Cutoff Frequency", 1, 250, 30)
        f_type = st.selectbox("Filter Type", ["Butterworth", "Gaussian", "Ideal"])
        
        res_freq = frequency_hpf(img_gray, f_cutoff, type=f_type)
        
        cf1, cf2 = st.columns(2)
        cf1.image(img_gray, caption="Original", use_container_width=True)
        cf2.image(res_freq, caption=f"{f_type} HPF (Cutoff={f_cutoff})", use_container_width=True)
        
        st.warning("⚠️ **Note**: Standard High-Pass filters often darken the image because they remove the DC component (average brightness). Use 'High-Frequency Emphasis' to keep the brightness.")

    # Summary Matrix
    st.divider()
    st.markdown("### 📋 Sharpening Comparison Matrix")
    st.table({
        "Method": ["Laplacian", "Unsharp Masking", "High-Pass Filter"],
        "Operation": ["2nd Derivative", "f + k*(f - blur)", "Frequency Cutoff"],
        "Strength": ["Isotropic (Edge direction independent)", "Highly controllable", "Natural results (Gaussian/Butterworth)"],
        "Risk": ["Amplifies noise", "Can cause halos at high k", "Can cause ringing (Ideal)"]
    })
