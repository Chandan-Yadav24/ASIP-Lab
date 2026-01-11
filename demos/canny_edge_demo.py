# demos/canny_edge_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import canny
from scipy.ndimage import gaussian_filter, sobel
import io

@st.cache_data
def load_img_canny(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic shapes with noise
        img_np = np.zeros((512, 512), dtype=np.uint8)
        img_np[100:412, 100:412] = 180 # Square
        y, x = np.ogrid[:512, :512]
        mask = (x - 256)**2 + (y - 256)**2 <= 80**2
        img_np[mask] = 80
        # Add some noise to justify the smoothing step
        noise = np.random.normal(0, 15, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)

def run():
    st.header("✨ Canny Edge Detector: The Gold Standard")
    st.markdown("""
    The Canny Edge Detector is an **optimal** multi-stage algorithm designed for:
    1. **Good Detection**: Maximize true edges, minimized false edges.
    2. **Good Localization**: Edges are exactly where they should be.
    3. **Single Response**: One thin line per edge (No thick bands).
    """)

    with st.expander("📚 Theory: The 5-Stage Pipeline", expanded=False):
        st.markdown(r"""
        ### Step 1: Gaussian Smoothing ($I_{smooth} = I * G_\sigma$)
        Suppresses high-frequency noise that causes false edges. Parameter $\sigma$ controls the scale.

        ### Step 2: Gradient Computation ($\nabla I$)
        Calculates intensity change magnitude $M$ and direction $\theta$ (usually via Sobel).
        
        ### Step 3: Non-Maximum Suppression (NMS)
        Thins edges to 1-pixel width. Checks if pixel is local max along gradient direction.

        ### Step 4: Double Thresholding
        Classifies pixels into:
        - **Strong** ($M > High$): Definitely an edge.
        - **Weak** ($Low \le M \le High$): Maybe an edge.
        - **Non-Edge** ($M < Low$): Discarded.

        ### Step 5: Edge Tracking by Hysteresis
        Keeps **Weak** edges only if they are connected to **Strong** edges. This preserves continuous contours while removing isolated noise.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="canny_local")
    img_gray = load_img_canny(local_up.read() if local_up else None)

    tab_pipeline, tab_hysteresis, tab_compare = st.tabs(["🚀 The Pipeline", "🧲 Hysteresis Lab", "🆚 Sobel vs Canny"])

    with tab_pipeline:
        st.subheader("Visualizing the Stages")
        
        sigma_p = st.slider("Step 1: Smoothing Sigma", 0.0, 5.0, 1.5, key="sig_p")
        
        # Step 1: Smooth
        if sigma_p > 0:
            smoothed = gaussian_filter(img_gray, sigma=sigma_p)
        else:
            smoothed = img_gray
            
        # Step 2: Gradient (Sobel)
        sx = sobel(smoothed, axis=1)
        sy = sobel(smoothed, axis=0)
        mag = np.hypot(sx, sy)
        mag_disp = (mag / mag.max() * 255).astype(np.uint8)
        
        # Step 3/4/5: Canny
        # We use skimage implementation for the final result
        edges_final = canny(img_gray, sigma=sigma_p)
        
        c1, c2, c3 = st.columns(3)
        c1.image(smoothed, caption="1. Smoothed Input", use_container_width=True)
        c2.image(mag_disp, caption="2. Gradient Magnitude", use_container_width=True)
        c3.image(edges_final.astype(np.uint8)*255, caption="5. Final Canny Output", use_container_width=True)
        
        st.info("Input -> Smooth -> Gradient -> NMS/Hysteresis -> Clean edges")

    with tab_hysteresis:
        st.subheader("Interactive Hysteresis")
        st.markdown("Adjust the **Low** and **High** thresholds to see how weak edges are linked to strong ones.")
        
        col_ctrl, col_view = st.columns([1, 2])
        
        with col_ctrl:
            sigma_h = st.slider("Sigma", 0.0, 5.0, 1.0, key="sig_h")
            low_t = st.slider("Low Threshold", 0, 255, 20)
            high_t = st.slider("High Threshold", 0, 255, 60)
            
            if low_t >= high_t:
                st.warning("⚠️ Low Threshold should be < High Threshold")

        with col_view:
             edges_h = canny(img_gray, sigma=sigma_h, low_threshold=low_t, high_threshold=high_t)
             st.image(edges_h.astype(np.uint8)*255, caption=f"Canny ($\sigma={sigma_h}, L={low_t}, H={high_t}$)", use_container_width=True)
             
        st.markdown("""
        - **High Threshold**: Controls which edges are "Strong" (Sure things).
        - **Low Threshold**: Controls which "Weak" edges are allowed to connect to Strong ones.
        - **Gap**: If Low is too low, noise gets attached. If High is too high, contours break.
        """)

    with tab_compare:
        st.subheader("Why Canny is Better")
        
        c_sob, c_can = st.columns(2)
        
        # Simple Sobel Magnitude Thresholding
        sob_thresh = st.slider("Sobel Threshold", 0, 255, 50, key="sob_cmp")
        sx_raw = sobel(img_gray, axis=1)
        sy_raw = sobel(img_gray, axis=0)
        m_raw = np.hypot(sx_raw, sy_raw)
        sob_edges = (m_raw > sob_thresh)
        
        with c_sob:
            st.image(sob_edges.astype(np.uint8)*255, caption="Standard Sobel (Magnitude > T)", use_container_width=True)
            st.caption("Issues: Thick edges, noise speckles, disconnected lines.")
            
        with c_can:
            # Matching Canny roughly
            can_edges = canny(img_gray, sigma=1.0, low_threshold=20, high_threshold=sob_thresh)
            st.image(can_edges.astype(np.uint8)*255, caption="Canny Edge Detector", use_container_width=True)
            st.caption("Benefits: Thin (1-px) edges, clean background, continuous contours.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **The Canny Edge Detector** is an optimal algorithm aiming for:
    1. **Good Detection** (Low error rate)
    2. **Good Localization** (Edges match reality)
    3. **Single Response** (Thin lines)
    
    **It uses 5 steps**: Gaussian Smooth → Gradient Calculation → Non-Max Suppression → Double Thresholding → Hysteresis Tracking.
    """)
