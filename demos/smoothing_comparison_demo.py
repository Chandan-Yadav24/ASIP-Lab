# demos/smoothing_comparison_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
import io

@st.cache_data
def load_img_smooth(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A clean image that we can add noise to
        # Creating a synthetic image with sharp edges and shapes
        img_np = np.zeros((512, 512), dtype=np.uint8)
        img_np[100:412, 100:412] = 200 # Bright square
        img_np[200:312, 200:312] = 100 # Inset square
        # Add a circle
        y, x = np.ogrid[:512, :512]
        mask = (x - 400)**2 + (y - 400)**2 <= 60**2
        img_np[mask] = 150
        img = Image.fromarray(img_np)
    return np.array(img)

@st.cache_data
def add_noise(img, noise_type, amount):
    if amount == 0: return img
    
    noisy = img.astype(float)
    if noise_type == "Gaussian":
        noise = np.random.normal(0, amount * 25, img.shape)
        noisy += noise
    elif noise_type == "Salt & Pepper":
        # amount is probability [0, 1]
        prob = amount * 0.2 # Scale it for reasonable noise
        mask = np.random.random(img.shape)
        noisy[mask < prob/2] = 0
        noisy[mask > 1 - prob/2] = 255
        
    return np.clip(noisy, 0, 255).astype(np.uint8)

def run():
    st.header("🌫️ Smoothing: Linear vs. Non-linear")
    st.markdown("""
    Smoothing reduces noise and detail. But **how** it does it matters—some filters blur everything (Linear), while others protect edges (Non-linear).
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: Frosted Glass vs. Smart Editor", expanded=False):
        st.markdown(r"""
        ### 1. Linear Smoothing (Convolution)
        Linear filters (Average, Gaussian) replace pixels with a **weighted sum** of neighbors.
        - **Mental Model**: *Frosted Glass*. Everything becomes a hazy blur.
        - **Low-Pass Filter**: Removes High frequencies (noise & sharp edges).
        
        ### 2. Non-linear Smoothing (Order-Statistics)
        Non-linear filters (Median) **sort** neighbors and pick one.
        - **Mental Model**: *Smart Editor*. Outliers (noise) are replaced by the most common neighbor.
        - **Edge Preservation**: Keeps sharp transitions much better than averaging.

        ### 3. Comparison Matrix
        | Filter | Logic | Best For | Trade-off |
        | :--- | :--- | :--- | :--- |
        | **Gaussian** | Weighted Average | Natural Blur | Blurs edges heavily |
        | **Median** | Order Ranking | Salt & Pepper | Computationally heavier |
        | **Bilateral** | Intensity Aware | High-end Denoise | Complexity |
        """)

    # --- Global Input Area ---
    st.sidebar.markdown("### 📥 Image & Noise Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="smooth_local")
    
    img_clean = load_img_smooth(local_up.read() if local_up else None)
    
    noise_type = st.sidebar.selectbox("Inject Noise", ["None", "Gaussian", "Salt & Pepper"])
    noise_amt = st.sidebar.slider("Noise Intensity", 0.0, 1.0, 0.2 if noise_type != "None" else 0.0)
    
    img_noisy = add_noise(img_clean, noise_type, noise_amt)
    
    st.info(f"📁 **Active Image**: {'Uploaded' if local_up else 'Synthetic Shapes'} | Noise: **{noise_type}** ({noise_amt*100:.0f}%)")

    tab_compare, tab_zoom, tab_matrix = st.tabs(["🧪 Filter Comparison Lab", "🔍 Edge-Preservation Zoom", "📋 Technical Matrix"])

    with tab_compare:
        st.subheader("Comparative Denoising")
        
        k_size = st.slider("Kernel Size (Radius)", 1, 15, 3)
        # Convert radius to actual size for median/uniform
        actual_size = k_size * 2 + 1
        
        col1, col2, col3 = st.columns(3)
        
        # 1. Average (Linear)
        img_avg = uniform_filter(img_noisy, size=actual_size)
        col1.image(img_avg, caption=f"Average Filter ({actual_size}x{actual_size})", use_container_width=True)
        col1.caption("Linear Blur")
        
        # 2. Gaussian (Linear)
        img_gauss = gaussian_filter(img_noisy, sigma=k_size/2)
        col2.image(img_gauss, caption=f"Gaussian Filter (σ={k_size/2:.1f})", use_container_width=True)
        col2.caption("Natural Smooth")
        
        # 3. Median (Non-linear)
        img_med = median_filter(img_noisy, size=actual_size)
        col3.image(img_med, caption=f"Median Filter ({actual_size}x{actual_size})", use_container_width=True)
        col3.caption("Non-Linear Selection")

        st.success("✅ **Observation**: Try 'Salt & Pepper' noise. Notice how Median deletes the white/black dots perfectly while Average just 'blurs' them!")

    with tab_zoom:
        st.subheader("The Edge Preservation Test")
        st.markdown("Zooming into a sharp transition to see the 'Edge Smearing' effect.")
        
        z_size = 100
        cx, cy = 250, 250 # Focus on edges
        
        # Slices
        zoom_orig = img_noisy[cy:cy+z_size, cx:cx+z_size]
        zoom_avg = img_avg[cy:cy+z_size, cx:cx+z_size]
        zoom_med = img_med[cy:cy+z_size, cx:cx+z_size]
        
        cz1, cz2, cz3 = st.columns(3)
        cz1.image(zoom_orig, caption="Noisy Edge", use_container_width=True)
        cz2.image(zoom_avg, caption="Blurred Edge (Linear)", use_container_width=True)
        cz3.image(zoom_med, caption="Sharp Edge (Non-linear)", use_container_width=True)

    with tab_matrix:
        st.subheader("📋 Filter Comparison Matrix")
        st.table({
            "Property": ["Type", "Edge Handling", "Noise Best Suited For", "Phase Distortion"],
            "Linear (Avg/Gauss)": ["Weighted Sum", "Heavy Blur", "White/Gaussian Noise", "Minimal"],
            "Non-linear (Median)": ["Order Statistics", "Preservation", "Impulse (Salt & Pepper)", "Signal dependent"]
        })
        
    st.divider()
    st.markdown("### 📋 Smoothing Profile")
    st.image(img_noisy, caption="Input Data (Noisy)", use_container_width=True)
