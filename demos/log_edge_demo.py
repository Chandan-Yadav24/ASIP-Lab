# demos/log_edge_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, gaussian_laplace
import io

@st.cache_data
def load_img_log(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic blobs for LoG demo
        img_np = np.zeros((512, 512), dtype=np.uint8)
        # Large blob
        y, x = np.ogrid[:512, :512]
        mask1 = (x - 150)**2 + (y - 150)**2 <= 60**2
        img_np[mask1] = 200
        # Medium blob
        mask2 = (x - 350)**2 + (y - 200)**2 <= 40**2
        img_np[mask2] = 180
        # Small blob
        mask3 = (x - 300)**2 + (y - 400)**2 <= 20**2
        img_np[mask3] = 220
        img = Image.fromarray(img_np)
    return np.array(img)

def run():
    st.header("🎩 Laplacian of Gaussian (LoG): The Mexican Hat")
    st.markdown("""
    LoG combines **Gaussian Smoothing** (noise reduction) with the **Laplacian** (2nd derivative edge detection).
    It's also known as the **Mexican Hat** operator due to its shape.
    """)

    with st.expander("📚 Theory: How LoG Works", expanded=False):
        st.markdown(r"""
        ### 1. The Idea
        - **Problem**: Laplacian alone amplifies noise.
        - **Solution**: Smooth first (Gaussian), then Laplacian.
        - **Result**: $\text{LoG} = \nabla^2 (G_\sigma * I) = (\nabla^2 G_\sigma) * I$

        ### 2. Zero Crossings
        At edges, the LoG response changes sign (+ to - or vice versa).
        - Finding these **zero crossings** gives thin, 1-pixel edges.
        - This is the basis of **Marr-Hildreth** edge detection.

        ### 3. DoG Approximation
        $$\text{DoG} = G_{\sigma_1} - G_{\sigma_2} \approx \text{LoG}$$
        Faster to compute, used in SIFT feature detection.

        ### 4. Scale-Space
        Varying $\sigma$ detects features at different scales:
        - Small $\sigma$: Fine details, small blobs.
        - Large $\sigma$: Coarse structures, large blobs.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="log_local")
    img_gray = load_img_log(local_up.read() if local_up else None)

    tab_kernel, tab_zero, tab_dog, tab_scale = st.tabs(["🎩 The Kernel", "🔀 Zero Crossings", "🔄 DoG Approx", "📏 Scale Space"])

    with tab_kernel:
        st.subheader("Mexican Hat Visualizer")
        
        sigma_k = st.slider("Kernel Sigma", 1.0, 10.0, 3.0, key="sig_k")
        
        # Generate 2D LoG kernel for visualization
        size = int(6 * sigma_k) | 1  # Odd size
        x = np.linspace(-size//2, size//2, size)
        y = np.linspace(-size//2, size//2, size)
        X, Y = np.meshgrid(x, y)
        
        # LoG formula
        log_kernel = -(1 / (np.pi * sigma_k**4)) * (1 - (X**2 + Y**2) / (2 * sigma_k**2)) * np.exp(-(X**2 + Y**2) / (2 * sigma_k**2))
        
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, log_kernel, cmap='coolwarm')
        ax1.set_title("LoG Kernel (Mexican Hat)")
        
        ax2 = fig.add_subplot(122)
        ax2.imshow(log_kernel, cmap='coolwarm')
        ax2.set_title("2D View")
        ax2.axis('off')
        
        st.pyplot(fig)
        st.info("Notice the central peak surrounded by a ring of opposite sign - the 'brim' of the hat.")

    with tab_zero:
        st.subheader("Edge Detection via Zero Crossings")
        
        sigma_z = st.slider("LoG Sigma", 1.0, 10.0, 2.0, key="sig_z")
        
        # Apply LoG
        log_response = gaussian_laplace(img_gray.astype(float), sigma=sigma_z)
        
        # Find zero crossings (sign changes between neighbors)
        zero_cross = np.zeros_like(log_response, dtype=np.uint8)
        for i in range(1, log_response.shape[0]-1):
            for j in range(1, log_response.shape[1]-1):
                neighbors = [log_response[i-1,j], log_response[i+1,j], log_response[i,j-1], log_response[i,j+1]]
                if log_response[i,j] > 0:
                    if any(n < 0 for n in neighbors):
                        zero_cross[i,j] = 255
                elif log_response[i,j] < 0:
                    if any(n > 0 for n in neighbors):
                        zero_cross[i,j] = 255
        
        c1, c2 = st.columns(2)
        c1.image((np.abs(log_response)/np.abs(log_response).max()*255).astype(np.uint8), caption="LoG Response (Magnitude)", use_container_width=True)
        c2.image(zero_cross, caption="Zero Crossings (Edges)", use_container_width=True)

    with tab_dog:
        st.subheader("Difference of Gaussians (DoG)")
        st.markdown("A fast approximation: $DoG = G_{\\sigma_1} - G_{\\sigma_2}$")
        
        s1 = st.slider("Sigma 1 (Smaller)", 0.5, 5.0, 1.0, key="dog_s1")
        s2 = st.slider("Sigma 2 (Larger)", 1.0, 10.0, 2.0, key="dog_s2")
        
        if s1 >= s2:
            st.warning("Sigma 1 should be < Sigma 2")
        
        g1 = gaussian_filter(img_gray.astype(float), sigma=s1)
        g2 = gaussian_filter(img_gray.astype(float), sigma=s2)
        dog = g1 - g2
        
        # Compare with true LoG
        true_log = gaussian_laplace(img_gray.astype(float), sigma=(s1+s2)/2)
        
        d1, d2 = st.columns(2)
        d1.image((np.abs(dog)/np.abs(dog).max()*255).astype(np.uint8), caption=f"DoG ($\\sigma_1={s1}, \\sigma_2={s2}$)", use_container_width=True)
        d2.image((np.abs(true_log)/np.abs(true_log).max()*255).astype(np.uint8), caption="True LoG", use_container_width=True)
        
        st.success("DoG is computationally cheaper and gives very similar results!")

    with tab_scale:
        st.subheader("Multi-Scale Blob Detection")
        
        sigmas = [2, 4, 8, 16]
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i, s in enumerate(sigmas):
            log_s = gaussian_laplace(img_gray.astype(float), sigma=s)
            axes[i].imshow(np.abs(log_s), cmap='hot')
            axes[i].set_title(f"$\\sigma = {s}$")
            axes[i].axis('off')
        
        st.pyplot(fig)
        st.info("Different $\\sigma$ values respond to blobs of different sizes. Small $\\sigma$ ➜ small blobs, large $\\sigma$ ➜ large blobs.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **LoG (Laplacian of Gaussian)** = Smoothing + 2nd Derivative.
    - **Zero Crossings**: Thin edge detection (Marr-Hildreth).
    - **DoG**: Fast approximation ($G_{\\sigma_1} - G_{\\sigma_2}$).
    - **Scale Space**: Vary $\\sigma$ for multi-scale analysis.
    """)
