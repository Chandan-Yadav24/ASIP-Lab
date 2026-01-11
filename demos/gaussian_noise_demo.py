# demos/gaussian_noise_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import stats
from . import dsp_utils

def run():
    st.subheader("Gaussian Noise: The Bell Curve in Images")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Gaussian (Normal) Noise", expanded=False):
        st.markdown(
            r"""
            **Gaussian Noise**: Random noise following a normal (bell curve) distribution.
            
            ### 1) Statistical Definition
            Fully described by two parameters:
            - **Mean (μ)**: Center/average value
            - **Standard Deviation (σ)**: Spread/strength of noise
            
            **Probability Density Function**:
            $$p(z) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)$$
            
            ### 2) Distribution Properties
            - **~68%** of values within μ ± σ
            - **~95%** of values within μ ± 2σ
            - **~99.7%** of values within μ ± 3σ
            
            ### 3) Why Gaussian?
            - **Common in Reality**: Sensor noise, electronic noise, thermal noise
            - **Central Limit Theorem**: Sum of many small effects → Gaussian
            - **Mathematical Convenience**: Clean properties for analysis
            
            ### 4) Image Processing Impact
            **Appearance**: Fine grain/speckled texture across image
            
            **Common Sources**:
            - Sensor noise
            - Low light conditions
            - High temperature
            
            **Edge Detection Problem**: Derivatives amplify noise, creating false edges
            
            ### 5) Noise Reduction
            - **Low-pass Filtering**: Mean/Gaussian blur (reduces noise but blurs edges)
            - **Adaptive Filtering**: Stronger smoothing in flat regions, weaker near edges
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Distribution Parameters")
        
        mu = st.slider("Mean (μ)", -1.0, 1.0, 0.0, 0.1, key="gaussian_mu")
        sigma = st.slider("Std Deviation (σ)", 0.1, 1.0, 0.3, 0.05, key="gaussian_sigma")
        
        st.info(f"μ = {mu:.2f}, σ = {sigma:.2f}")
        
        # Generate samples
        n_samples = 10000
        samples = np.random.normal(mu, sigma, n_samples)
        
        st.write("### 2. Image Noise Demo")
        
        noise_strength = st.slider("Image Noise Strength", 0.0, 0.5, 0.1, 0.05, key="gaussian_img_noise")
        
        # Create test pattern (checkerboard)
        size = 128
        pattern = np.zeros((size, size))
        pattern[::16, :] = 1
        pattern[:, ::16] = 1
        
        # Add Gaussian noise
        noisy_image = pattern + np.random.normal(0, noise_strength, pattern.shape)
        noisy_image = np.clip(noisy_image, 0, 1)
        
        st.write("### 3. Filtering")
        
        filter_type = st.selectbox(
            "Filter Type",
            ["None", "Mean Filter (3x3)", "Gaussian Blur (σ=1)"],
            key="gaussian_filter"
        )
        
        if filter_type == "Mean Filter (3x3)":
            filtered_image = ndimage.uniform_filter(noisy_image, size=3)
        elif filter_type == "Gaussian Blur (σ=1)":
            filtered_image = ndimage.gaussian_filter(noisy_image, sigma=1.0)
        else:
            filtered_image = noisy_image
        
        st.write("### 4. Listen to Gaussian Noise")
        
        audio_duration = st.slider("Audio Duration (s)", 0.5, 3.0, 1.0, key="gaussian_audio_duration")
        
        # Generate Gaussian noise audio
        framerate = 11025
        audio_signal = dsp_utils.UncorrelatedGaussianNoise(amp=0.3)
        audio_wave = audio_signal.make_wave(duration=audio_duration, framerate=framerate)
        
        st.audio(audio_wave.get_audio_bytes(), format='audio/wav')
        st.caption("White Gaussian noise sounds like static/hiss")

    with col_viz:
        st.write("### Analysis")
        
        # A) Histogram with Bell Curve
        st.markdown("**A) Distribution: Histogram vs Theoretical Bell Curve**")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Histogram
        counts, bins, patches = ax.hist(samples, bins=50, density=True, 
                                        alpha=0.7, color='skyblue', edgecolor='black')
        
        # Theoretical curve
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        theoretical = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, theoretical, 'r-', linewidth=2, label=f'Theoretical: N({mu:.1f}, {sigma:.2f}²)')
        
        # Mark μ±σ, μ±2σ, μ±3σ
        ax.axvline(mu, color='green', linestyle='--', alpha=0.5, label='μ')
        ax.axvline(mu + sigma, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(mu - sigma, color='orange', linestyle='--', alpha=0.5, label='μ±σ')
        ax.axvline(mu + 2*sigma, color='red', linestyle='--', alpha=0.3)
        ax.axvline(mu - 2*sigma, color='red', linestyle='--', alpha=0.3, label='μ±2σ')
        
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.set_title("Gaussian Distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        
        # Statistics
        within_1sigma = np.sum((samples >= mu - sigma) & (samples <= mu + sigma)) / n_samples * 100
        within_2sigma = np.sum((samples >= mu - 2*sigma) & (samples <= mu + 2*sigma)) / n_samples * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Within μ±σ", f"{within_1sigma:.1f}%", delta="Expected: 68%")
        col2.metric("Within μ±2σ", f"{within_2sigma:.1f}%", delta="Expected: 95%")
        
        # B) Image Comparison
        st.markdown("**B) Image Processing: Noise Addition & Filtering**")
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(pattern, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Original Pattern")
        axes[0].axis('off')
        
        axes[1].imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Noisy (σ={noise_strength:.2f})")
        axes[1].axis('off')
        
        axes[2].imshow(filtered_image, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f"Filtered ({filter_type})")
        axes[2].axis('off')
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # C) Edge Detection Impact
        st.markdown("**C) Edge Detection: Impact of Noise**")
        
        # Sobel edge detection
        edges_clean = np.abs(ndimage.sobel(pattern))
        edges_noisy = np.abs(ndimage.sobel(noisy_image))
        edges_filtered = np.abs(ndimage.sobel(filtered_image))
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(edges_clean, cmap='hot')
        axes[0].set_title("Edges (Clean)")
        axes[0].axis('off')
        
        axes[1].imshow(edges_noisy, cmap='hot')
        axes[1].set_title("Edges (Noisy) - False Edges!")
        axes[1].axis('off')
        
        axes[2].imshow(edges_filtered, cmap='hot')
        axes[2].set_title("Edges (Filtered)")
        axes[2].axis('off')
        
        fig.tight_layout()
        st.pyplot(fig)
        
        st.warning("⚠️ **Observation**: Noise creates many false edges because derivatives amplify random variations!")
