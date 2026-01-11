# demos/sift_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color, exposure, data, util
from PIL import Image
import io

@st.cache_data
def load_img_sift(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Using the astronaut image for complex feature matching visualization
        img_np = data.astronaut()
        img_np = color.rgb2gray(img_np)
    return img_np

def run():
    st.header("🛡️ SIFT: Scale-Invariant Feature Transform")
    st.markdown("""
    SIFT (David Lowe, 1999) is a legendary algorithm that finds points in an image that can be matched even if the object is rotated, scaled, or lighted differently.
    """)

    with st.expander("📚 Theory: The 4 Stages of SIFT", expanded=False):
        st.markdown(r"""
        ### 1. Scale-space Extrema Detection
        Search over all scales and image locations using a **Difference of Gaussians (DoG)** pyramid to find potential interest points.
        
        ### 2. Keypoint Localization
        Refine the location and scale. Discard points with low contrast or points that are poorly localized along edges.
        
        ### 3. Orientation Assignment
        Assign one or more orientations to each keypoint based on local image gradient directions. This achieves **Rotation Invariance**.
        
        ### 4. Keypoint Descriptor
        Compute a **128-dimensional** vector representing the local neighborhood. This "fingerprint" is used for matching.
        
        ### 🧩 The Mosaic Analogy
        Finding a patterned tile in a mosaic:
        - **Scale-space**: Looking from different heights to find patterns that stay visible.
        - **Orientation**: Marking which way the pattern points so it matches even if rotated.
        - **Descriptor**: A detailed fingerprint that uniquely identifies the tile.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="sift_local")
    img = load_img_sift(local_up.read() if local_up else None)
    img_gray = exposure.rescale_intensity(img.astype(float))

    st.sidebar.markdown("### 🧰 Parameters")
    n_keypoints = st.sidebar.slider("Number of Keypoints (Approx)", 50, 1000, 200)

    tab_viz, tab_dog, tab_vector = st.tabs(["🎯 Keypoint Visualizer", "🪜 Scale-Space (DoG)", "🧬 Descriptor Explorer"])

    # Note: Using SIFT from skimage.feature
    # In some versions of skimage, SIFT might be SIFT() or daisy()
    # We will use the built-in SIFT extractor
    detector_extractor = feature.SIFT(n_octaves=3, n_scales=3, sigma_min=1.6)
    detector_extractor.detect_and_extract(img_gray)
    
    keypoints = detector_extractor.keypoints[:n_keypoints]
    scales = detector_extractor.scales[:n_keypoints]
    orientations = detector_extractor.orientations[:n_keypoints]
    descriptors = detector_extractor.descriptors[:n_keypoints]

    with tab_viz:
        st.subheader("Scale and Orientation Invariance")
        st.markdown("Circles represent the detected **Scale**, and lines represent the **Orientation**.")
        
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        
        # Plot keypoints with scale and orientation
        for i in range(len(keypoints)):
            y, x = keypoints[i]
            s = scales[i]
            o = orientations[i]
            
            # Draw circle for scale
            circle = plt.Circle((x, y), s*3, color='cyan', fill=False, linewidth=1, alpha=0.6)
            ax.add_patch(circle)
            
            # Draw orientation arrow
            dx = np.cos(o) * s * 3
            dy = np.sin(o) * s * 3
            ax.arrow(x, y, dx, dy, color='yellow', head_width=2, alpha=0.8)
            
        ax.set_title(f"Detected {len(keypoints)} SIFT Keypoints")
        ax.axis('off')
        st.pyplot(fig)
        st.success("By measuring the local orientation, the descriptor can be 'normalized' to always face up, making it rotation invariant!")

    with tab_dog:
        st.subheader("Difference of Gaussians (DoG)")
        st.markdown("SIFT finds keypoints by subtracting two blurred versions of the image.")
        
        s1 = st.slider("Sigma 1", 0.5, 10.0, 1.6)
        s2 = st.slider("Sigma 2", 0.5, 10.0, 3.2)
        
        blur1 = filters.gaussian(img, sigma=s1)
        blur2 = filters.gaussian(img, sigma=s2)
        dog = blur2 - blur1
        
        col1, col2 = st.columns(2)
        col1.image(np.clip(exposure.rescale_intensity(dog, out_range=(0, 1)), 0, 1), caption="DoG (Subtracted Blurs)", use_container_width=True)
        col2.markdown("""
        **Why DoG?**
        - It is a fast approximation of the **Laplacian of Gaussian (LoG)**.
        - Extrema in this map indicate structures that exist at this specific scale.
        """)

    with tab_vector:
        st.subheader("The 128-D Fingerprint")
        st.markdown("Each keypoint is converted into a 128-dimensional vector (4x4 subregions x 8 orientations).")
        
        if len(descriptors) > 0:
            idx = st.number_input("Select Keypoint Index", 0, len(descriptors)-1, 0)
            
            st.write(f"**Descriptor Vector for Keypoint {idx}:**")
            st.bar_chart(descriptors[idx])
            
            st.info("This vector is normalized so it doesn't change much if the image gets brighter or darker.")
        else:
            st.warning("No keypoints detected.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **SIFT**: Scale-Invariant Feature Transform.
    - **Invariance**: Handles Scale, Rotation, and Illumination.
    - **Key Steps**: DoG for Scale Space $\rightarrow$ Filter unstable points $\rightarrow$ Assign Orientation $\rightarrow$ 128-D Vector.
    - **Matching**: Ratio Test ($d_1 / d_2 < 0.8$) is crucial to remove ambiguous matches.
    """)

# Helper imports
from skimage import filters
