# demos/feature_extraction_intro_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color, data, util, exposure
from PIL import Image
import io

@st.cache_data
def load_img_features(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Use a rich image from skimage
        img_np = data.checkerboard()
        # Mix in some other textures
        img_np[100:150, 100:150] = data.coins()[0:50, 0:50] 
    return img_np

def run():
    st.header("⚡ Feature Extraction: Detection & Description")
    st.markdown("""
    Feature extraction is a mid-level process that converts raw pixels into meaningful **keypoints** (where things are) and **descriptors** (what those things look like).
    """)

    with st.expander("📚 Theory: Where vs What", expanded=False):
        st.markdown(r"""
        ### 1. Feature Detection (Where)
        Detectors locate distinctive points that are stable across different views.
        - **Corners**: Points with high intensity variation in all directions (e.g., Harris).
        - **Blobs**: Regions that are lighter or darker than surroundings (e.g., LoG).
        
        ### 2. Feature Description (What)
        Once a point is found, we describe its neighborhood using a numerical vector.
        - **Desired Properties**: Invariance to scale, rotation, and illumination.
        - **Examples**: SIFT (Histogram of gradients), BRIEF (Binary tests).
        
        ### 3. Summary Table
        | Component | Main Job | Output | Examples |
        |---|---|---|---|
        | **Detector** | Find points (Where) | Coordinates | Harris, FAST |
        | **Descriptor** | Encode neighborhood (What) | Feature Vectors | HOG, BRIEF |
        | **Combined** | Both | Keypoints + Vectors | SIFT, ORB |
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="feat_local")
    img = load_img_features(local_up.read() if local_up else None)

    tab_detect, tab_combined, tab_analogy = st.tabs(["🎯 Feature Detectors", "⚡ Combined Frameworks (ORB)", "🕵️ Summary & Analogy"])

    with tab_detect:
        st.subheader("Locating Distinctive Points")
        detector_type = st.selectbox("Choose Detector:", ["Harris Corners", "Canny Edges (Low-level)", "Blob Detector (LoG)"])
        
        if detector_type == "Harris Corners":
            sigma = st.slider("Harris Sigma (Smoothing)", 0.1, 5.0, 1.0)
            coords = feature.corner_peaks(feature.corner_harris(img, sigma=sigma), min_distance=5, threshold_rel=0.02)
            
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.plot(coords[:, 1], coords[:, 0], '+r', markersize=10, label='Detected Corners')
            ax.set_title("Harris Corner Detection")
            ax.axis('off')
            st.pyplot(fig)
            st.info("Corners are stable interest points because they have strong gradients in multiple directions.")
            
        elif detector_type == "Blob Detector (LoG)":
            max_sigma = st.slider("Max Sigma (Blob Size)", 5, 50, 30)
            blobs = feature.blob_log(img, max_sigma=max_sigma, num_sigma=10, threshold=.1)
            
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            for blob in blobs:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
                ax.add_patch(c)
            ax.set_title("Laplacian of Gaussian (LoG) Blobs")
            ax.axis('off')
            st.pyplot(fig)
            st.info("Blob detectors find regions that stand out from the background at different scales.")
            
        else: # Canny
            edges = feature.canny(img, sigma=1.5)
            st.image(edges, caption="Canny Edges", use_container_width=True)

    with tab_combined:
        st.subheader("ORB: Oriented FAST and Rotated BRIEF")
        st.markdown("""
        ORB is a fast, efficient alternative to SIFT. It **detects** keypoints (FAST) and **describes** them (BRIEF) with rotation invariance.
        """)
        
        n_keypoints = st.slider("Number of Keypoints", 10, 500, 100)
        
        detector_extractor = feature.ORB(n_keypoints=n_keypoints)
        detector_extractor.detect_and_extract(img)
        keypoints = detector_extractor.keypoints
        descriptors = detector_extractor.descriptors
        
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.scatter(keypoints[:, 1], keypoints[:, 0], s=20, edgecolors='r', facecolors='none', label='ORB Keypoints')
        ax.set_title(f"{len(keypoints)} ORB Keypoints Detected")
        ax.axis('off')
        st.pyplot(fig)
        
        st.success(f"Extracted {len(keypoints)} keypoints. Each point has a 256-bit descriptor vector.")
        
        with st.expander("👁️ View Sample Descriptors (What the computer sees)"):
            if len(descriptors) > 0:
                st.write(f"Sample Descriptor (Point 0):")
                st.code(descriptors[0].astype(int))
                st.caption("Each '1' or '0' represents a binary test in the keypoint's neighborhood.")

    with tab_analogy:
        st.subheader("The Investigator Analogy")
        st.info("""
        **Investigation in a Crowd**:
        - **Detection**: Spotting a unique face landmark (e.g., mole, sharp chin) → Tells you **where** to look.
        - **Description**: Recording measurable details (shape, size, pattern) → Tells you **what** it is.
        - **Matching**: Comparing your notes to recognize the same person under different lighting or viewpoints.
        """)
        st.markdown("""
        #### Why Invariance Matters?
        If a descriptor isn't **invariant to rotation**, it will fail to recognize the same corner if the camera is tilted!
        """)

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Feature Detection**: Localization ($X, Y$). Stable points across views.
    - **Feature Description**: Encoding neighborhood ($N$-dim vector). Comparison/Matching.
    - **Key Desired Property**: **Invariance** (Scale, Rotation, Illumination).
    - **Popular Frameworks**: SIFT, SURF (Classic/Patented), **ORB** (Fast/Free).
    """)
