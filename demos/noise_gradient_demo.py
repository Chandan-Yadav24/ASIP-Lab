# demos/noise_gradient_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
import io

@st.cache_data
def load_img_noise_grad(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic clean step/ramp image
        img_np = np.zeros((512, 512), dtype=np.uint8)
        img_np[:, :256] = 50
        img_np[:, 256:] = 200 # Sharp vertical edge
        # Add a ramp at the bottom
        for r in range(400, 512):
            img_np[r, :] = np.linspace(0, 255, 512)
        img = Image.fromarray(img_np)
    return np.array(img)

def run():
    st.header("⚡ Noise vs. Gradients: The High-Frequency Problem")
    st.markdown("""
    Derivatives are **high-pass filters**. They love rapid changes. Unfortunately, **Noise** is also a rapid change. 
    This leads to "Gradient Explosion" where noise can overwhelm real edges.
    """)

    with st.expander("📚 Theory: Why Derivatives Hate Noise", expanded=False):
        st.markdown(r"""
        ### 1. Frequency Response
        - Derivative response $\propto j\omega$. 
        - As frequency $\omega$ increases (fine details & noise), the response grows linearly.
        - **Result**: High-frequency noise gets amplified massively compared to low-frequency signals.

        ### 2. The Problems
        - **Amplification**: Small random fluctuations becoming huge spikes.
        - **False Edges**: Noise peaks looking like real edges.
        - **Masking**: Real edges getting buried in the chaos.

        ### 3. The Solution: Pre-Smoothing
        Smooth *before* you differentiate.
        - **Step 1**: Gaussian Blur (suppress high $\omega$).
        - **Step 2**: Compute Gradient.
        - This effectively created a **Band-Pass** filter that targets "Edge-sized" frequencies.

        ### 4. Analogy: Path in a Storm 🌪️
        - **Edge**: The path you want to follow.
        - **Noise**: The storm whipping grass back and forth.
        - If you look at every blade of grass (Derivative), you see chaos.
        - If you **squint** (Smoothing), the random motion blurs out, and the steady path remains visible.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="noise_grad_local")
    img_clean = load_img_noise_grad(local_up.read() if local_up else None)

    # Global Noise injection for standard demo consistency
    noise_sigma = st.sidebar.slider("Noise Level (Sigma)", 0.0, 50.0, 15.0)
    
    # 1D Image slice for analysis
    row_idx = st.sidebar.slider("Select Row for 1D Analysis", 0, img_clean.shape[0]-1, img_clean.shape[0]//2)

    img_noisy = img_clean.astype(float) + np.random.normal(0, noise_sigma, img_clean.shape)
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

    tab_1d, tab_2d, tab_matrix = st.tabs(["📈 1D Chaos Lab", "🔳 2D Cleanup Lab", "📋 Summary Matrix"])

    with tab_1d:
        st.subheader("1D Gradient Explosion")
        
        # Extract profiles
        prof_clean = img_clean[row_idx, :].astype(float)
        prof_noisy = img_noisy[row_idx, :].astype(float)
        
        # Derivatives
        deriv_clean = np.gradient(prof_clean)
        deriv_noisy = np.gradient(prof_noisy)
        
        # Smoothed Derivative
        smooth_sigma = st.slider("Smoothing Sigma (1D)", 0.5, 10.0, 3.0, key="s1")
        prof_smooth = gaussian_filter(prof_noisy, sigma=smooth_sigma)
        deriv_smooth = np.gradient(prof_smooth)
        
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        
        # Row 1: Signals
        axes[0].plot(prof_clean, 'k--', alpha=0.5, label='Clean Signal')
        axes[0].plot(prof_noisy, 'r-', alpha=0.6, label='Noisy Signal')
        axes[0].plot(prof_smooth, 'b-', lw=2, label=f'Smoothed ($\sigma={smooth_sigma}$)')
        axes[0].set_title("Intensity Profiles")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Row 2: Noisy Derivative
        axes[1].plot(deriv_clean, 'k--', alpha=0.5, label="Clean Edge (Target)")
        axes[1].plot(deriv_noisy, 'r-', alpha=0.8, label="Noisy Gradient")
        axes[1].set_title("Result 1: Gradient of Noisy Signal (Explosion!)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Row 3: Smoothed Derivative
        axes[2].plot(deriv_clean, 'k--', alpha=0.5, label="Clean Edge (Target)")
        axes[2].plot(deriv_smooth, 'b-', lw=2, label="Smoothed Gradient")
        axes[2].set_title("Result 2: Gradient of Smoothed Signal (Restored)")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        st.pyplot(fig)
        st.info("Observe how the 'Noisy Gradient' has peaks everywhere (False Edges), while the 'Smoothed Gradient' recovers the single true peak.")

    with tab_2d:
        st.subheader("2D Edge Detection: Sobel vs. Noise")
        
        c1, c2 = st.columns(2)
        
        # Naive Sobel on Noisy Image
        sobel_x_noisy = sobel(img_noisy, axis=1)
        sobel_y_noisy = sobel(img_noisy, axis=0)
        mag_noisy = np.hypot(sobel_x_noisy, sobel_y_noisy)
        mag_noisy = (mag_noisy / mag_noisy.max() * 255).astype(np.uint8)
        
        # Pre-smoothed Sobel
        s_sigma_2d = st.slider("Smoothing Sigma (2D)", 0.5, 10.0, 2.0, key="s2")
        
        smooth_2d = gaussian_filter(img_noisy, sigma=s_sigma_2d)
        sobel_x_smooth = sobel(smooth_2d, axis=1)
        sobel_y_smooth = sobel(smooth_2d, axis=0)
        mag_smooth = np.hypot(sobel_x_smooth, sobel_y_smooth)
        mag_smooth = (mag_smooth / mag_smooth.max() * 255).astype(np.uint8)

        with c1:
            st.image(mag_noisy, caption="Gradient of Noisy Input", use_container_width=True)
            st.caption("Lots of speckles (False Edges).")
            
        with c2:
            st.image(mag_smooth, caption=f"Gradient of Smoothed Input ($\sigma={s_sigma_2d}$)", use_container_width=True)
            st.caption("Clean edges restored.")

    with tab_matrix:
        st.subheader("Effects Summary")
        st.table({
            "Effect": ["HF Amplification", "Edge Masking", "False Edges", "Magnitude Inflation"],
            "Mechanism": ["Derivative ~ Frequency", "Noise peaks > Signal peaks", "Random noise gradients", "Larger jumps = Stronger response"],
            "Result": ["Noise Boost", "Lost features", "Spurious detections", "Global clutter"]
        })
        st.markdown("### 💡 The Golden Rule")
        st.success("**Always Smooth Before You Differentiate!**")
