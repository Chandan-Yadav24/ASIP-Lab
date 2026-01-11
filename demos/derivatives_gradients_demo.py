# demos/derivatives_gradients_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import sobel, laplace, gaussian_filter
import io

@st.cache_data
def load_img_deriv(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic shapes for clean edge detection
        img_np = np.zeros((512, 512), dtype=np.uint8)
        img_np[100:412, 100:412] = 200 # Square
        y, x = np.ogrid[:512, :512]
        mask = (x - 256)**2 + (y - 256)**2 <= 80**2
        img_np[mask] = 50 # Dark circle inside
        img = Image.fromarray(img_np)
    return np.array(img)

def run():
    st.header("📉 Derivatives: The Engine of Edge Detection")
    st.markdown("""
    If smoothing is integration (blurring), edge detection is **differentiation**. We look for rapid changes in intensity.
    """)

    with st.expander("📚 Theory: Gradient vs. Laplacian", expanded=False):
        st.markdown(r"""
        ### 1. First-Order Derivative: The Gradient ($\nabla f$)
        - **Vector**: $\nabla f = [g_x, g_y]^T$ pointing to max intensity increase.
        - **Magnitude**: $M(x,y) = \sqrt{g_x^2 + g_y^2}$ (Edge Strength).
        - **Logic**: "Hand Velocity" - Detects the *speed* of intensity change.

        ### 2. Second-Order Derivative: The Laplacian ($\nabla^2 f$)
        - **Scalar**: Isotropic operator (directionless).
        - **Logic**: "Sudden Jolt" - Reacts to the start/end of a ramp.
        - **Zero-Crossing**: The distinct sign change (+ to -) marks the exact edge location.

        ### 3. Noise Sensitivity
        Derivatives amplify noise. 
        - **Solution**: **LoG** (Laplacian of Gaussian) - Smooth first, then differentiate!
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="deriv_local")
    img_gray = load_img_deriv(local_up.read() if local_up else None)
    
    tab_1d, tab_2d, tab_log = st.tabs(["📈 1D Signal Lab", "🔳 2D Gradient Lab", "🌊 LoG vs Noise"])

    with tab_1d:
        st.subheader("Feeling the Staircase (1D Analysis)")
        
        # Synthetic 1D Signal
        x = np.linspace(0, 100, 200)
        y = np.zeros_like(x)
        y[50:100] = np.linspace(0, 100, 50) # Ramp
        y[100:150] = 100 # Constant
        y[150:200] = 50 # Step down
        
        # Derivatives
        dy = np.gradient(y)
        d2y = np.gradient(dy)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        
        ax1.plot(x, y, 'k-', lw=2)
        ax1.set_title("Original Signal f(x): Ramp & Step")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(x, dy, 'b-', lw=2)
        ax2.fill_between(x, dy, alpha=0.2, color='blue')
        ax2.set_title("1st Derivative f'(x): Velocity (Non-zero on Ramp)")
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(x, d2y, 'r-', lw=2)
        ax3.set_title("2nd Derivative f''(x): Acceleration (Jolt at ends of Ramp)")
        ax3.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        st.info("Notice how f''(x) is zero on the ramp but spikes where the ramp starts and stops!")

    with tab_2d:
        st.subheader("2D Edge Detection")
        
        c1, c2 = st.columns(2)
        
        # Sobel Gradients
        sobel_x = sobel(img_gray, axis=1) # Horizontal changes
        sobel_y = sobel(img_gray, axis=0) # Vertical changes
        magnitude = np.hypot(sobel_x, sobel_y)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        
        with c1:
            st.image(np.abs(sobel_x).astype(np.uint8), caption="Sobel X (Vertical Edges)", use_container_width=True)
            st.image(np.abs(sobel_y).astype(np.uint8), caption="Sobel Y (Horizontal Edges)", use_container_width=True)
            
        with c2:
            st.image(magnitude, caption="Gradient Magnitude (Combined)", use_container_width=True)
            # Laplacian
            lap = np.abs(laplace(img_gray)).astype(np.uint8)
            # Enhance visibility
            lap = (lap / lap.max() * 255).astype(np.uint8)
            st.image(lap, caption="Laplacian (Double Edges)", use_container_width=True)

    with tab_log:
        st.subheader("The Noise Problem")
        
        noise_amt = st.slider("Add Noise", 0.0, 50.0, 10.0)
        noisy_img = img_gray + np.random.normal(0, noise_amt, img_gray.shape)
        
        sigma = st.slider("Gaussian Sigma (LoG)", 0.5, 5.0, 2.0)
        
        # Raw Laplacian on Noisy Image
        raw_lap = np.abs(laplace(noisy_img))
        raw_lap = (raw_lap / raw_lap.max() * 255).astype(np.uint8)
        
        # LoG: Smooth then Lap
        smooth = gaussian_filter(noisy_img, sigma=sigma)
        log_res = np.abs(laplace(smooth))
        log_res = (log_res / log_res.max() * 255).astype(np.uint8)
        
        l1, l2 = st.columns(2)
        l1.image(raw_lap, caption="Raw Laplacian (Noise Amplified!)", use_container_width=True)
        l2.image(log_res, caption=f"LoG (Sigma={sigma})", use_container_width=True)
        
        st.success("✅ **Insight**: Without smoothing, the derivative just highlights the noise. LoG fixes this.")

    st.divider()
    st.markdown("### 📋 Derivatives Comparison")
    st.table({
        "Type": ["Gradient (1st)", "Laplacian (2nd)", "LoG"],
        "Output": ["Vector (Mag + Dir)", "Scalar", "Scalar"],
        "Features": ["Detects edge presence", "Detects fine detail (Double response)", "Robust to noise"],
        "Analogy": ["Hand Velocity", "Sudden Jolt", "Jolt with gloves on"]
    })
