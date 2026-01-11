# demos/pca_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from skimage import color, data, util
from PIL import Image
import io

@st.cache_data
def load_img_pca(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Using a rich image for compression demo
        img_np = data.camera()
    return img_np

def perform_pca_on_image(img, n_components):
    # Standardize data: Subtract mean
    mean = np.mean(img, axis=0)
    centered_img = img - mean
    
    # Compute Covariance Matrix
    # C = (Z.T @ Z) / (n - 1)
    cov_matrix = np.cov(centered_img, rowvar=False)
    
    # Eigen-decomposition
    evals, evecs = linalg.eigh(cov_matrix)
    
    # Sort by descending eigenvalues
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    # Select top k components
    selected_evecs = evecs[:, :n_components]
    
    # Project data
    projected = centered_img @ selected_evecs
    
    # Reconstruct
    reconstructed = (projected @ selected_evecs.T) + mean
    
    return reconstructed, evals, projected

def run():
    st.header("📊 Principal Component Analysis (PCA)")
    st.markdown("""
    PCA is a dimensionality reduction technique that finds the directions of **maximum variance** in high-dimensional data.
    """)

    with st.expander("📚 Theory: Math of Variance", expanded=False):
        st.markdown(r"""
        ### 1. The Core Idea
        PCA transforms correlated variables into a set of values of linearly uncorrelated variables called **Principal Components (PCs)**.
        - **PC1**: Direction of highest variance.
        - **PC2**: Next highest variance, perpendicular (orthogonal) to PC1.
        
        ### 2. Algorithm Steps
        1. **Standardize**: Subtract mean to center the data.
        2. **Covariance**: Compute $C = \frac{1}{n-1} Z^T Z$.
        3. **Eigen-analysis**: Find Eigenvectors ($V$) and Eigenvalues ($\lambda$) of $C$.
        4. **Select**: Keep top $k$ components based on sorted eigenvalues.
        5. **Project**: $Z_{reduced} = Z V_k$.
        
        ### 3. Usage in Images
        - **Hotelling Transform**: Another name for PCA in signal processing.
        - **Compression**: Throw away low-variance components to save space.
        - **Denoising**: Noise lives in low-variance components; discarding them cleans the signal.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="pca_local")
    img = load_img_pca(local_up.read() if local_up else None)

    tab_2d, tab_compress, tab_variance = st.tabs(["📉 2D Projection Lab", "🏗️ Image Compression", "📊 Variance Analysis"])

    with tab_2d:
        st.subheader("Visualizing Principal Axes")
        st.markdown("Taking pixels from two rows and showing how PCA finds the principal direction.")
        
        # Pick two rows for 2D visualization
        row1 = img[100, :].astype(float)
        row2 = img[150, :].astype(float)
        data_2d = np.vstack([row1, row2]).T # Shape (N, 2)
        
        # PCA on 2D
        mean_2d = np.mean(data_2d, axis=0)
        centered_2d = data_2d - mean_2d
        cov_2d = np.cov(centered_2d, rowvar=False)
        evals_2d, evecs_2d = linalg.eigh(cov_2d)
        
        fig, ax = plt.subplots()
        ax.scatter(centered_2d[:, 0], centered_2d[:, 1], alpha=0.3, label='Centered Data')
        
        # Plot Principal Axes
        for i in range(2):
            length = np.sqrt(evals_2d[i]) * 2 # Scale by std dev
            v = evecs_2d[:, i] * length
            ax.arrow(0, 0, v[0], v[1], color='red', head_width=5, linewidth=2)
        
        ax.set_xlabel("Pixel Intensity (Row 100)")
        ax.set_ylabel("Pixel Intensity (Row 150)")
        ax.set_title("Data Spread & Principal Components")
        ax.legend()
        st.pyplot(fig)
        st.info("The red arrows show the principal components (axes of variation). The longest arrow is PC1.")

    with tab_compress:
        st.subheader("Image Compression via PCA")
        st.markdown("Approximating the image using only a fraction of its principal components.")
        
        max_comps = img.shape[1]
        n_comps = st.slider("Number of Principal Components", 1, max_comps, 20)
        
        reconstructed, _, _ = perform_pca_on_image(img.astype(float), n_comps)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original Image", use_container_width=True)
        col2.image(reconstructed, caption=f"Reconstructed (k={n_comps})", use_container_width=True)
        
        st.success(f"With only {n_comps} components, we preserve the most important structural information.")

    with tab_variance:
        st.subheader("Scree Plot: Explained Variance")
        _, evals, _ = perform_pca_on_image(img.astype(float), max_comps)
        
        total_var = np.sum(evals)
        explained_variance_ratio = evals / total_var
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        fig, ax = plt.subplots()
        ax.plot(range(1, 51), explained_variance_ratio[:50], 'o-', label="Individual")
        ax.step(range(1, 51), cumulative_variance[:50], where='mid', label="Cumulative")
        ax.set_xlabel("Components")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Top 50 Principal Components")
        ax.legend()
        st.pyplot(fig)
        
        comp_90 = np.argmax(cumulative_variance >= 0.90) + 1
        st.write(f"**90% of variance** is captured by the first **{comp_90}** components.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Principal Component Analysis (PCA)**:
    - **Orthogonal**: New axes are uncorrelated.
    - **Variance**: PC1 captures the most, then PC2, etc.
    - **Compression**: Store top $k$ components instead of all $N$ variables.
    - **Denoising**: Effectively discards low-variance 'noise' components.
    - **Terminology**: Eigenvectors are the 'directions', Eigenvalues are the 'strengths'.
    """)
