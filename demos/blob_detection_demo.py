# demos/blob_detection_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color, data, util, exposure
from PIL import Image
import io

@st.cache_data
def load_img_blobs(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Hubbel image or coins/stars are classic for blob detection
        img_np = data.coins()
    return img_np

def run():
    st.header("🔭 Blob Detection: Finding Regions of Interest")
    st.markdown("""
    Blob detection identifies regions that differ in intensity or texture from their surroundings. 
    It is the key to finding stars in the sky, cells in a microscope, or coins on a table.
    """)

    with st.expander("📚 Theory: LoG, DoG, and DoH", expanded=False):
        st.markdown(r"""
        ### 1. Laplacian of Gaussian (LoG)
        The most accurate method. It blurs the image at multiple scales ($\sigma$) and finds local maxima in the "Scale Space".
        - **Accuracy**: High
        - **Speed**: Slow (computes 3D extrema)
        
        ### 2. Difference of Gaussian (DoG)
        A fast approximation of LoG. Instead of calculating derivatives, it subtracts two differently blurred images.
        - **Accuracy**: Medium
        - **Speed**: Fast
        
        ### 3. Determinant of Hessian (DoH)
        Uses the Hessian matrix of second-order partial derivatives. Its speed is independent of blob size.
        - **Accuracy**: Medium
        - **Speed**: Fastest (constant time per scale)
        
        ### 🌌 The Telescope Analogy
        - **Stars** are like blobs: bright points against a dark background.
        - **LoG** is like a high-precision lens that finds each star's center and size (scale) exactly.
        - **DoH** is like a fast scanner that counts light sources quickly, but might miss tiny specks.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="blob_local")
    img = load_img_blobs(local_up.read() if local_up else None)
    img_gray = exposure.rescale_intensity(img.astype(float))

    st.sidebar.markdown("### 🧰 Parameters")
    min_sigma = st.sidebar.slider("Min Sigma (Smallest Blob)", 1, 10, 1)
    max_sigma = st.sidebar.slider("Max Sigma (Largest Blob)", 10, 100, 30)
    threshold = st.sidebar.slider("Detector Threshold", 0.01, 1.0, 0.2)

    tab_log, tab_comparison, tab_table = st.tabs(["🎯 LoG (Precision)", "🆚 Algorithm Comparison", "📊 Summary & Properties"])

    with tab_log:
        st.subheader("Laplacian of Gaussian (LoG) Lab")
        st.markdown("Searching for 3D extrema in scale-space.")
        
        blobs_log = feature.blob_log(img_gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=10, threshold=threshold)
        # Compute radii: sqrt(2) * sigma
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
        
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        for blob in blobs_log:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
            ax.add_patch(c)
        ax.set_title(f"LoG: Detected {len(blobs_log)} Blobs")
        ax.axis('off')
        st.pyplot(fig)
        st.info("Yellow circles indicate the detected center and estimated scale (size) of each blob.")

    with tab_comparison:
        st.subheader("LoG vs DoG vs DoH")
        col1, col2, col3 = st.columns(3)
        
        # 1. LoG (Already computed)
        with col1:
            st.write("**LoG** (Accurate)")
            st.image(img, use_container_width=True)
            st.caption(f"Count: {len(blobs_log)}")
            
        # 2. DoG
        blobs_dog = feature.blob_dog(img_gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
        with col2:
            st.write("**DoG** (Fast Approx)")
            fig2, ax2 = plt.subplots()
            ax2.imshow(img, cmap='gray')
            for blob in blobs_dog:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='lime', linewidth=1, fill=False)
                ax2.add_patch(c)
            ax2.axis('off')
            st.pyplot(fig2)
            st.caption(f"Count: {len(blobs_dog)}")

        # 3. DoH
        blobs_doh = feature.blob_doh(img_gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        with col3:
            st.write("**DoH** (Fastest)")
            fig3, ax3 = plt.subplots()
            ax3.imshow(img, cmap='gray')
            for blob in blobs_doh:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='cyan', linewidth=1, fill=False)
                ax3.add_patch(c)
            ax3.axis('off')
            st.pyplot(fig3)
            st.caption(f"Count: {len(blobs_doh)}")
            
        st.warning("Notice how DoH is much faster but may cluster nearby blobs together differently compared to LoG.")

    with tab_table:
        st.subheader("Algorithm Comparison Matrix")
        st.table({
            "Method": ["LoG", "DoG", "DoH"],
            "Core Idea": ["Gaussian + Laplacian", "Difference of Gaussians", "Hessian Determinant"],
            "Speed": ["Slow", "Medium", "Fastest"],
            "Accuracy": ["High", "Medium", "Medium"],
            "Best For": ["Precise localization", "Fast approximation", "Large-scale real-time"]
        })
        
        st.markdown("""
        ### Why this Matters?
        Blob detection is the foundation for:
        - **Medical Imaging**: Detecting anomalous cells or tissues.
        - **Astronomy**: Locating stars and galaxies.
        - **Tracking**: Following objects (blobs) across frames in a video.
        """)

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Blob**: A region noticeably different from its neighbors.
    - **LoG**: The gold standard for precision.
    - **Scale Space**: Searching at multiple blurs ($\sigma$) to find blobs of different sizes.
    - **DoG**: Rapidly approximates LoG by subtracting blurs.
    - **DoH**: Best for speed, identifies both bright and dark regional extrema.
    """)
