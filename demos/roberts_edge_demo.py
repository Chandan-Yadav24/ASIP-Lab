# demos/roberts_edge_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
import io

@st.cache_data
def load_img_roberts(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic clean diamond shape to test diagonal sensitivity
        img_np = np.zeros((512, 512), dtype=np.uint8)
        # Draw a rotated square (diamond)
        y, x = np.ogrid[:512, :512]
        # |x| + |y| <= R
        mask = (np.abs(x - 256) + np.abs(y - 256)) <= 150
        img_np[mask] = 200
        # Add a normal square too
        img_np[50:150, 50:150] = 150
        img = Image.fromarray(img_np)
    return np.array(img)

def run():
    st.header("⚡ Roberts Cross Operator: The Diagonal Specialist")
    st.markdown("""
    One of the earliest and simplest edge detectors. It uses tiny **2x2 kernels** to check for diagonal changes.
    """)

    with st.expander("📚 Theory: Roberts Operator", expanded=False):
        st.markdown(r"""
        ### 1. The 2x2 Kernels
        Approximates gradient derivatives at 45° and 135°.
        $$
        G_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad
        G_y = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}
        $$

        ### 2. Gradient Calculation
        For a 2x2 block $\begin{bmatrix} p_1 & p_2 \\ p_3 & p_4 \end{bmatrix}$:
        - $G_x = p_1 - p_4$ (Top-Left minus Bottom-Right)
        - $G_y = p_2 - p_3$ (Top-Right minus Bottom-Left)
        
        Magnitude: $M \approx \sqrt{G_x^2 + G_y^2}$ or $|G_x| + |G_y|$

        ### 3. Pros & Cons
        - **Pro**: Extremely fast (only 4 pixels). Good for diagonal edges.
        - **Con**: **Very Noise Sensitive** (No smoothing!). High false detection rate.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="roberts_local")
    img_gray = load_img_roberts(local_up.read() if local_up else None)

    tab_calc, tab_vis, tab_noise = st.tabs(["🧮 Numerical Example", "🔳 Visual Lab", "⚠️ Noise Sensitivity"])

    with tab_calc:
        st.subheader("Interactive 2x2 Calculation")
        st.markdown("Try the numbers from the example or your own!")
        
        c1, c2 = st.columns(2)
        with c1:
            p1 = st.number_input("P1 (Top-Left)", 0, 255, 10)
            p3 = st.number_input("P3 (Btm-Left)", 0, 255, 15)
        with c2:
            p2 = st.number_input("P2 (Top-Right)", 0, 255, 50)
            p4 = st.number_input("P4 (Btm-Right)", 0, 255, 200)
            
        gx_val = p1 - p4
        gy_val = p2 - p3
        mag_exact = np.sqrt(gx_val**2 + gy_val**2)
        mag_approx = abs(gx_val) + abs(gy_val)
        
        st.markdown("---")
        st.markdown(f"**G_x** = {p1} - {p4} = **{gx_val}**")
        st.markdown(f"**G_y** = {p2} - {p3} = **{gy_val}**")
        st.markdown(f"**Magnitude** (Exact) = $\sqrt{{ ({gx_val})^2 + ({gy_val})^2 }} \\approx$ **{mag_exact:.2f}**")
        st.markdown(f"**Magnitude** (Approx) = $|{gx_val}| + |{gy_val}| =$ **{mag_approx}**")
        
        if mag_exact > 50:
            st.success("Strong Edge Detected!")
        else:
            st.warning("Weak/No Edge.")

    with tab_vis:
        st.subheader("Roberts in Action")
        
        # Roberts kernels
        # Note: Scipy convolve flips kernel, so for proper correlation we flip back or define rotated
        # Technically Roberts is defined on +1/2 coordinates, discrete approx usually shifts.
        # We will use the standard definitions for implementation.
        
        rx = np.array([[1, 0], [0, -1]])
        ry = np.array([[0, 1], [-1, 0]])
        
        gx_img = convolve(img_gray.astype(float), rx)
        gy_img = convolve(img_gray.astype(float), ry)
        
        mag_img = np.hypot(gx_img, gy_img)
        mag_disp = (mag_img / mag_img.max() * 255).astype(np.uint8)
        
        c1, c2, c3 = st.columns(3)
        c1.image(np.abs(gx_img).astype(np.uint8), caption="Roberts X (Diagonal 1)", use_container_width=True)
        c2.image(np.abs(gy_img).astype(np.uint8), caption="Roberts Y (Diagonal 2)", use_container_width=True)
        c3.image(mag_disp, caption="Magnitude", use_container_width=True)
        
        st.info("Check how the *Diamond shape* edges are highlighted differently by X and Y compared to the Square.")

    with tab_noise:
        st.subheader("The Downside: Noise")
        
        n_sigma = st.slider("Add Noise", 0, 50, 20, key="rob_noise")
        noisy = img_gray.astype(float) + np.random.normal(0, n_sigma, img_gray.shape)
        
        # Roberts on Noisy
        r_mag = np.hypot(convolve(noisy, rx), convolve(noisy, ry))
        
        # Sobel on Noisy (for comparison)
        from scipy.ndimage import sobel
        s_mag = np.hypot(sobel(noisy, axis=1), sobel(noisy, axis=0))
        
        n1, n2 = st.columns(2)
        n1.image((r_mag/r_mag.max()*255).astype(np.uint8), caption="Roberts (No Smoothing)", use_container_width=True)
        n2.image((s_mag/s_mag.max()*255).astype(np.uint8), caption="Sobel (Implicit Smoothing)", use_container_width=True)
        
        st.warning("Notice how much grittier the Roberts output is. It amplifies every little noise speck.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Roberts Operator** uses 2x2 masks to find diagonal gradients.
    - **Pros**: Simplest, Fastest.
    - **Cons**: High noise sensitivity (amplifies noise), poor localization compared to Canny.
    """)
