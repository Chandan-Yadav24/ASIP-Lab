# demos/sobel_edge_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
import io

@st.cache_data
def load_img_sobel(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic shapes for clean edge detection
        img_np = np.zeros((512, 512), dtype=np.uint8)
        img_np[100:412, 100:412] = 200 # Square
        y, x = np.ogrid[:512, :512]
        mask = (x - 256)**2 + (y - 256)**2 <= 80**2
        img_np[mask] = 50 # Dark circle inside
        # Add a softer edge blur initially to show the gradient response better
        from scipy.ndimage import gaussian_filter
        img_np = gaussian_filter(img_np, sigma=1.0)
        img = Image.fromarray(img_np)
    return np.array(img)

def plot_kernel(kernel, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(kernel, cmap='coolwarm', vmin=-2, vmax=2)
    for (i, j), z in np.ndenumerate(kernel):
        ax.text(j, i, f'{z}', ha='center', va='center', fontsize=14, weight='bold')
    ax.set_title(title)
    ax.axis('off')
    return fig

def run():
    st.header("⚡ Sobel Operator: The Standard Edge Detector")
    st.markdown("""
    The Sobel operator detects edges by finding where intensity changes rapidly. It's the industry standard for basic edge detection because it combines **Differentiation** (finding changes) with **Smoothing** (noise reduction).
    """)

    with st.expander("📚 Theory: How Sobel Works", expanded=False):
        st.markdown(r"""
        ### 1. Operational Mechanism
        Sobel approximates the gradient (derivative) using two 3x3 kernels:
        - **$G_x$ (Horizontal)**: Responds to vertical edges (Top/Bottom are same, Left/Right differ).
        - **$G_y$ (Vertical)**: Responds to horizontal edges (Left/Right are same, Top/Bottom differ).
        
        ### 2. Gradient Magnitude
        $G = \sqrt{G_x^2 + G_y^2}$
        - Large G: Strong Edge.
        - Orientation: $\theta = \arctan(G_y / G_x)$

        ### 3. Built-In Smoothing
        Unique feature: Sobel is effectively a **Smoothing Filter** in one direction $\times$ **Derivative Filter** in the other.
        - Example $G_x$: Smoothing $[1, 2, 1]^T$ $\times$ Difference $[-1, 0, 1]$.
        - This makes it robust against noise compared to simple finite differences.

        ### 4. Applications
        - **Edge Detection**: Finding object outlines.
        - **Feature Extraction**: Input for detecting shapes (lanes, faces).
        - **Segmentation**: Marking boundaries between regions.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="sobel_local")
    img_gray = load_img_sobel(local_up.read() if local_up else None)
    
    tab_kernels, tab_edges, tab_thresh = st.tabs(["🎛️ Derivative Kernels", "🔳 Component Visualizer", "🎯 Edge Map & Thresholding"])

    with tab_kernels:
        st.subheader("The Convolution Kernels")
        
        c1, c2 = st.columns(2)
        
        hx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        hy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        with c1:
            st.pyplot(plot_kernel(hx, "Sobel-X (Vertical Edge Detector)"))
            st.caption("Checks Left vs Right. Center column is zero.")
            
        with c2:
            st.pyplot(plot_kernel(hy, "Sobel-Y (Horizontal Edge Detector)"))
            st.caption("Checks Top vs Bottom. Center row is zero.")

    with tab_edges:
        st.subheader("Separating X and Y Gradients")
        
        # Convolve
        gx = convolve(img_gray.astype(float), hx)
        gy = convolve(img_gray.astype(float), hy)
        
        # Magnitude
        mag = np.hypot(gx, gy)
        mag_disp = (mag / mag.max() * 255).astype(np.uint8)
        
        # Visualize Components
        col_x, col_y, col_m = st.columns(3)
        col_x.image(np.abs(gx).astype(np.uint8), caption="G_x (Vertical Edges)", use_container_width=True)
        col_y.image(np.abs(gy).astype(np.uint8), caption="G_y (Horizontal Edges)", use_container_width=True)
        col_m.image(mag_disp, caption="Gradient Magnitude (Combined)", use_container_width=True)
        
        st.info("Notice how the square's vertical sides light up in G_x, and horizontal sides light up in G_y.")

    with tab_thresh:
        st.subheader("Creating a Binary Edge Map")
        st.markdown("In real applications, we threshold the magnitude to classify 'Edge' vs 'Not Edge'.")
        
        thresh = st.slider("Gradient Threshold", 0, 255, 50)
        
        binary_edges = (mag > thresh).astype(np.uint8) * 255
        
        b1, b2 = st.columns(2)
        b1.image(mag_disp, caption="Raw Magnitude", use_container_width=True)
        b2.image(binary_edges, caption=f"Thresholded Edges (T > {thresh})", use_container_width=True)
        
        st.caption("Lower threshold = More details (but more noise). Higher threshold = Strongest edges only.")

    st.divider()
    st.markdown("### 📋 Sobel Profile")
    st.table({
        "Feature": ["Efficiency", "Noise Robustness", "Use Case"],
        "Details": ["Fast 3x3 convolution", "Includes Gaussian-like smoothing", "General purpose Edge Detection"]
    })
