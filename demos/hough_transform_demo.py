# demos/hough_transform_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform, color, data, util, exposure
from PIL import Image
import io

@st.cache_data
def load_img_hough(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Shapes image or building is classic for Hough line detection
        img_np = data.camera()
    return img_np

def run():
    st.header("📏 4.3.1 Hough Transform for Line Detection")
    st.markdown("""
    The Hough Transform is a robust method for detecting geometric shapes like lines and circles, even if they are broken or noisy.
    """)

    with st.expander("📚 Theory: The Normal Form & Voting", expanded=False):
        st.markdown(r"""
        ### 1. The Normal Representation
        Instead of $y = mx + c$, SIFT uses the **Normal Form**:
        $$x \cos \theta + y \sin \theta = \rho$$
        - **$\rho$ (Rho)**: Distance from origin to the line.
        - **$\theta$ (Theta)**: Angle of the perpendicular.
        
        ### 2. Hough Space (Accumulator)
        - **1 Point in Image** $\rightarrow$ **1 Sinusoidal Curve** in $(\rho, \theta)$ space.
        - **Intersection in Hough Space** $\rightarrow$ **1 Line** in Image space.
        
        ### 🔦 Torches in a Dark Field Analogy
        Imagine people holding torches in a dark field. To see if they line up:
        - Each person "votes" for every possible line they could belong to.
        - In the "Control Tower" (Accumulator), if many votes pile up for the same line coordinates, we know a real line exists!
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="hough_local")
    img = load_img_hough(local_up.read() if local_up else None)
    img_gray = util.img_as_float(img)

    st.sidebar.markdown("### 🧰 Parameters")
    canny_sigma = st.sidebar.slider("Canny Smoothing (Sigma)", 1.0, 5.0, 2.0)
    num_peaks = st.sidebar.slider("Number of Peaks (Lines)", 1, 50, 10)
    min_dist = st.sidebar.slider("Min Peak Distance (Hough Space)", 1, 50, 10)

    tab_accum, tab_recon, tab_table = st.tabs(["🗳️ Accumulator Space", "📏 Line Reconstruction", "📊 Summary Framework"])

    # Preprocessing
    edges = feature.canny(img_gray, sigma=canny_sigma)
    
    # Standard Hough Transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = transform.hough_line(edges, theta=tested_angles)

    with tab_accum:
        st.subheader("Hough Space: The Voting Board")
        st.markdown("Each bright spot here represents a detected line in the image.")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use log scale for better visibility of accumulator peaks
        h_vis = np.log(1 + h)
        ax.imshow(h_vis, extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                  cmap='hot', aspect=1/1.5)
        ax.set_title("Hough Accumulator (Parametric Space)")
        ax.set_xlabel("Theta (degrees)")
        ax.set_ylabel("Rho (pixels)")
        st.pyplot(fig)
        st.info("The x-axis represents the angle $\theta$ and the y-axis represents the distance $\rho$. Peaks (bright spots) correspond to dominant lines.")

    with tab_recon:
        st.subheader("Detected Lines Overlay")
        st.markdown("Reconstructing lines from the strongest accumulator peaks.")
        
        peaks_h, peaks_angles, peaks_dists = transform.hough_line_peaks(h, theta, d, num_peaks=num_peaks, min_distance=min_dist)
        
        fig_res, ax_res = plt.subplots()
        ax_res.imshow(img, cmap='gray')
        
        for angle, dist in zip(peaks_angles, peaks_dists):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax_res.axline((x0, y0), slope=np.tan(angle + np.pi/2), color='cyan', linewidth=1, alpha=0.7)
            
        ax_res.set_title(f"Top {len(peaks_h)} Detected Lines")
        ax_res.axis('off')
        st.pyplot(fig_res)
        st.success(f"Detected {len(peaks_h)} lines by finding local maxima in Hough Space.")

    with tab_table:
        st.subheader("Hough Line Framework Summary")
        st.table({
            "Concept": ["Normal Form", "Hough Space", "Point Mapping", "Peak Detection", "Resilience"],
            "Meaning": [
                "x·cosθ + y·sinθ = ρ",
                "(ρ, θ) coordinates",
                "1 Point → 1 Curve",
                "Accumulator local maxima",
                "Ignores gaps/noise"
            ],
            "Why it matters": [
                "Handles vertical lines easily",
                "Peak finding identifies shapes",
                "Captures all line possibilities",
                "Validates geometric evidence",
                "Works where local methods fail"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Normal Form**: Essential to handle vertical lines ($\theta=0$).
    - **Accumulator**: A 2D array representing discretized $(\rho, \theta)$ space.
    - **Voting**: Every edge pixel 'votes' for all lines it could belong to.
    - **Strength**: Global method; excellent for broken edges and noisy environments.
    """)
