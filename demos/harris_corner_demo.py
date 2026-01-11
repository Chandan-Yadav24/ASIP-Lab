# demos/harris_corner_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color, filters, util, data
from PIL import Image
import io
from scipy import ndimage

@st.cache_data
def load_img_harris(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Checkerboard is the classic corner detection image
        img_np = data.checkerboard()
    return img_np

def run():
    st.header("🎯 Harris Corner Detector: The Computer's Eye")
    st.markdown("""
    The Harris Corner Detector (1988) is a foundational algorithm used to find **stable interest points** (corners) where intensity changes sharply in all directions.
    """)

    with st.expander("📚 Theory: The Math of Shifting", expanded=False):
        st.markdown(r"""
        ### 1. The Structure Tensor (M)
        Harris looks at the weighted sum of squared gradients in a local window:
        $$M = \sum_{w} \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$
        - $I_x, I_y$: Image gradients (Sobel).
        - $\sum_w$: Spatial smoothing (usually Gaussian).
        
        ### 2. The Response Function (R)
        Instead of direct Eigenvalue computation, we use:
        $$R = \text{det}(M) - k(\text{tr}(M))^2$$
        - $\text{det}(M) = \lambda_1 \lambda_2$
        - $\text{tr}(M) = \lambda_1 + \lambda_2$
        - $k$: Sensitivity constant (usually 0.04 - 0.06).
        
        ### 3. Interpreting R
        - **$R > 0$ (Large)**: Both $\lambda$ large $\rightarrow$ **Corner**.
        - **$R < 0$ (Large)**: One $\lambda$ large $\rightarrow$ **Edge**.
        - **$|R| \approx 0$**: Both $\lambda$ small $\rightarrow$ **Flat Region**.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="harris_local")
    img = load_img_harris(local_up.read() if local_up else None)
    img_f = img.astype(float)

    st.sidebar.markdown("### 🧰 Parameters")
    k = st.sidebar.slider("Sensitivity (k)", 0.01, 0.1, 0.04, step=0.01)
    sigma = st.sidebar.slider("Smoothing (Sigma)", 0.5, 5.0, 1.0)

    tab_tensor, tab_response, tab_final = st.tabs(["🏗️ Structure Tensor", "🌡️ Response Map (R)", "🎯 Result Detector"])

    # Manual Tensor Components for Education
    Ix = ndimage.sobel(img_f, axis=1)
    Iy = ndimage.sobel(img_f, axis=0)
    
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy
    
    # Gaussian smoothing of components
    Sx2 = ndimage.gaussian_filter(Ix2, sigma)
    Sy2 = ndimage.gaussian_filter(Iy2, sigma)
    Sxy = ndimage.gaussian_filter(Ixy, sigma)

    with tab_tensor:
        st.subheader("Building the Harris Matrix Components")
        st.markdown("We look at how intensity changes in X, Y, and diagonals.")
        
        c1, c2, c3 = st.columns(3)
        c1.image(util.img_as_ubyte(exposure.rescale_intensity(Ix2, out_range=(0, 1))), caption="Ix² (Vertical Variance)", use_container_width=True)
        c2.image(util.img_as_ubyte(exposure.rescale_intensity(Iy2, out_range=(0, 1))), caption="Iy² (Horizontal Variance)", use_container_width=True)
        c3.image(util.img_as_ubyte(exposure.rescale_intensity(Ixy, out_range=(0, 1))), caption="Ixy (Cross Variance)", use_container_width=True)
        
        st.info("The smoothed versions of these maps form the 'Harris Matrix' at every pixel.")

    with tab_response:
        st.subheader("Harris Response Heatmap")
        st.markdown(r"$R = \det(M) - k \cdot \text{tr}(M)^2$")
        
        det = Sx2 * Sy2 - Sxy**2
        trace = Sx2 + Sy2
        R = det - k * (trace**2)
        
        # Colorize R for visibility (Pos=Red=Corner, Neg=Blue=Edge)
        fig, ax = plt.subplots()
        im = ax.imshow(R, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(im, label="Response R")
        ax.set_title(f"Harris Response Map (k={k})")
        ax.axis('off')
        st.pyplot(fig)
        
        st.warning("**Red regions** indicate high corner likelihood. **Blue regions** indicate edges.")

    with tab_final:
        st.subheader("Peak Detection & Non-Max Suppression")
        
        thresh_rel = st.slider("Response Threshold (Relative)", 0.0, 0.5, 0.05)
        min_dist = st.slider("Min Distance between Corners", 1, 20, 5)
        
        peaks = feature.corner_peaks(R, min_distance=min_dist, threshold_rel=thresh_rel)
        
        fig_res, ax_res = plt.subplots()
        ax_res.imshow(img, cmap='gray')
        ax_res.plot(peaks[:, 1], peaks[:, 0], 'r+', markersize=8, label='Keypoints')
        ax_res.set_title(f"Detected {len(peaks)} Corners")
        ax_res.axis('off')
        st.pyplot(fig_res)
        
        st.success("Corners are 'anchors' of an image—stable, unique, and easy to find from any angle.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Harris Corner Summary**:
    - **Invariance**: Rotation Invariant (eigenvalues don't change with tilt).
    - **Scale**: NOT Scale Invariant (a corner zoomed in looks like an edge).
    - **Eigen Pattern**: Corner ($\lambda_1, \lambda_2 \gg 0$), Edge ($\lambda_1 \gg \lambda_2 \approx 0$).
    - **Analogy**: Sliding a magnifying glass - if shifting any amount in any direction changes the view, it's a corner!
    """)

# Helper imports for image normalization
from skimage import exposure
