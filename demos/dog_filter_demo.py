# demos/dog_filter_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import io

@st.cache_data
def load_img_dog(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic blobs at different scales
        img_np = np.zeros((512, 512), dtype=np.uint8)
        y, x = np.ogrid[:512, :512]
        # Large blob
        mask1 = (x - 128)**2 + (y - 128)**2 <= 50**2
        img_np[mask1] = 200
        # Medium blob
        mask2 = (x - 350)**2 + (y - 150)**2 <= 30**2
        img_np[mask2] = 180
        # Small blob
        mask3 = (x - 256)**2 + (y - 400)**2 <= 15**2
        img_np[mask3] = 220
        # Add some texture
        img_np = img_np + np.random.randint(0, 20, img_np.shape).astype(np.uint8)
        img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
    return np.array(img)

def run():
    st.header("🔄 Difference of Gaussians (DoG): The Efficient Edge & Blob Detector")
    st.markdown("""
    DoG is a **band-pass filter** that approximates LoG but is much faster.
    Core idea: Blur twice at different scales, then subtract.
    """)

    with st.expander("📚 Theory: How DoG Works", expanded=False):
        st.markdown(r"""
        ### 1. The Formula
        $$D(x,y) = G_{\sigma_1} * I - G_{\sigma_2} * I = (G_{\sigma_1} - G_{\sigma_2}) * I$$
        where $\sigma_2 > \sigma_1$ (typically $\sigma_2 = k \cdot \sigma_1$, with $k \approx 1.6$).

        ### 2. Band-Pass Behavior
        - Both Gaussians remove high-frequency noise.
        - Subtraction removes low-frequency background.
        - **Result**: Mid-frequencies (edges, blobs) are enhanced.

        ### 3. LoG Approximation
        DoG $\approx$ Laplacian of Gaussian when $k$ is close to 1.
        This is why it produces similar zero-crossings.

        ### 4. SIFT Application
        Build a **DoG pyramid** across multiple scales. Local extrema in 3D (x, y, scale) are **keypoints** (scale-invariant blobs).
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="dog_local")
    img_gray = load_img_dog(local_up.read() if local_up else None)

    tab_1d, tab_visual, tab_pyramid = st.tabs(["📈 1D Signal Demo", "🔳 2D DoG Lab", "🏔️ Scale-Space Pyramid"])

    with tab_1d:
        st.subheader("1D DoG: Band-Pass in Action")
        
        # Create 1D signal with edge + noise
        x = np.linspace(0, 100, 500)
        signal = np.zeros_like(x)
        signal[150:350] = 1.0  # Step edge
        signal = signal + np.random.normal(0, 0.1, signal.shape)
        
        s1 = st.slider("Sigma 1 (Fine)", 1.0, 10.0, 2.0, key="1d_s1")
        s2 = st.slider("Sigma 2 (Coarse)", 2.0, 20.0, 5.0, key="1d_s2")
        
        g1 = gaussian_filter(signal, sigma=s1)
        g2 = gaussian_filter(signal, sigma=s2)
        dog_1d = g1 - g2
        
        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(x, signal, 'k-', alpha=0.5, label='Original + Noise')
        axes[0].plot(x, g1, 'b-', label=f'Blur $\\sigma_1={s1}$')
        axes[0].plot(x, g2, 'r-', label=f'Blur $\\sigma_2={s2}$')
        axes[0].legend()
        axes[0].set_title("Two Gaussian Blurs")
        
        axes[1].plot(x, dog_1d, 'g-', lw=2)
        axes[1].axhline(0, color='gray', alpha=0.5)
        axes[1].set_title("DoG = Blur1 - Blur2")
        
        axes[2].fill_between(x, dog_1d, alpha=0.3, color='green')
        axes[2].set_title("DoG Response (Band-Pass)")
        
        st.pyplot(fig)
        st.info("Notice: DoG responds at the *edges* of the step, not the flat regions (background removed).")

    with tab_visual:
        st.subheader("2D DoG Visualizer")
        
        s1_2d = st.slider("Sigma 1", 0.5, 5.0, 1.0, key="2d_s1")
        k = st.slider("k (Scale Ratio)", 1.2, 2.5, 1.6, key="2d_k")
        s2_2d = s1_2d * k
        
        g1_2d = gaussian_filter(img_gray.astype(float), sigma=s1_2d)
        g2_2d = gaussian_filter(img_gray.astype(float), sigma=s2_2d)
        dog_2d = g1_2d - g2_2d
        
        c1, c2, c3 = st.columns(3)
        c1.image((g1_2d / g1_2d.max() * 255).astype(np.uint8), caption=f"Blur $\\sigma_1={s1_2d:.1f}$", use_container_width=True)
        c2.image((g2_2d / g2_2d.max() * 255).astype(np.uint8), caption=f"Blur $\\sigma_2={s2_2d:.1f}$", use_container_width=True)
        
        # Normalize DoG for display
        dog_disp = dog_2d - dog_2d.min()
        dog_disp = (dog_disp / dog_disp.max() * 255).astype(np.uint8)
        c3.image(dog_disp, caption="DoG (Edges Enhanced)", use_container_width=True)

    with tab_pyramid:
        st.subheader("Scale-Space Pyramid (SIFT-like)")
        st.markdown("Build DoG responses at multiple scales to detect blobs of various sizes.")
        
        base_sigma = st.slider("Base Sigma", 0.5, 3.0, 1.0, key="pyr_base")
        k_pyr = 1.6
        
        sigmas = [base_sigma * (k_pyr ** i) for i in range(4)]
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            s1_p = sigmas[i]
            s2_p = s1_p * k_pyr
            g1_p = gaussian_filter(img_gray.astype(float), sigma=s1_p)
            g2_p = gaussian_filter(img_gray.astype(float), sigma=s2_p)
            dog_p = g1_p - g2_p
            
            axes[i].imshow(np.abs(dog_p), cmap='hot')
            axes[i].set_title(f"$\\sigma \\approx {s1_p:.1f}$")
            axes[i].axis('off')
        
        st.pyplot(fig)
        st.info("Blobs appear brightest at their 'natural' scale. Small blobs ➜ small $\\sigma$, large blobs ➜ large $\\sigma$.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **DoG** = Fast LoG approximation via $G_{\\sigma_1} - G_{\\sigma_2}$.
    - **Band-Pass**: Removes background + noise, keeps edges/blobs.
    - **Scale-Space**: Vary $\\sigma$ to detect features at multiple sizes.
    - **SIFT**: Uses DoG pyramid for scale-invariant keypoint detection.
    """)
