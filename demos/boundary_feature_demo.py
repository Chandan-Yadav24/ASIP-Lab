# demos/boundary_feature_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color, data, util, feature
from skimage.morphology import disk, binary_dilation
from PIL import Image
import io

@st.cache_data
def load_img_descriptors(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Create a synthetic image with distinct shapes
        img_np = np.zeros((300, 300), dtype=np.uint8)
        # Solid Rectangle
        img_np[50:120, 50:120] = 255
        # Circle with a hole
        yy, xx = np.mgrid[:300, :300]
        mask_outer = (yy - 200)**2 + (xx - 100)**2 <= 60**2
        mask_inner = (yy - 200)**2 + (xx - 100)**2 <= 20**2
        img_np[mask_outer] = 255
        img_np[mask_inner] = 0
        # Inclined Ellipse
        mask_ellipse = ((xx - 200)/60)**2 + ((yy - 100)/30)**2 <= 1
        img_np[mask_ellipse] = 255
    return img_np

def run():
    st.header("🔲 Boundary Processing & Feature Description")
    st.markdown("""
    After segmentation, mid-level processing describes objects using **Boundary** (outline) and **Regional** (area) descriptors.
    """)

    with st.expander("📚 Theory: Encoding the Pieces", expanded=False):
        st.markdown(r"""
        ### 1. Boundary Representation
        - **Tracing**: Ordering pixels to form a coherent contour (e.g., Moore tracing).
        - **Chain Codes**: Representing steps (0-7) in a neighborhood.
        - **Polygonal Approx**: Reducing a complex border to a few straight segments.
        
        ### 2. Descriptors (Attributes)
        - **Shape Metrics**: Area ($A$), Perimeter ($P$), Compactness ($P^2/A$), Eccentricity.
        - **Topological**: Euler Number ($E = C - H$, where $C$=components, $H$=holes).
        - **Texture**: Statistical measures of intensity like Mean, Variance, and Entropy.
        
        ### 🕵️ The Jigsaw Analogy
        - **Boundary Processing**: Tracing the jagged edge to get the piece outline.
        - **Feature Descriptors**: Writing notes about that outline (length, bumps, texture) to find a match quickly without testing every piece.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="desc_local")
    img = load_img_descriptors(local_up.read() if local_up else None)
    
    # Pre-process: Thresholding for region extraction
    thresh = st.sidebar.slider("Binary Threshold", 0, 255, 128)
    binary = img > thresh
    label_img = measure.label(binary)
    regions = measure.regionprops(label_img)

    tab_tracing, tab_shape, tab_topo, tab_texture = st.tabs(["➰ Boundary Tracing", "📏 Shape Metrics", "🔣 Topology", "🧱 Texture Analysis"])

    with tab_tracing:
        st.subheader("Boundary Extraction & Tracing")
        st.markdown("Finding the ordered sequence of coordinates along the edge.")
        
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        
        contours = measure.find_contours(binary, 0.5)
        for i, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, label=f"Object {i+1}")
        
        ax.set_title(f"Detected {len(contours)} Object Boundaries")
        ax.axis('off')
        st.pyplot(fig)
        
        if contours:
            st.write(f"Sample Boundary Points (Object 1): First 5 coordinates")
            st.write(contours[0][:5])

    with tab_shape:
        st.subheader("Interactive Shape Explorer")
        st.markdown("Extracting numerical attributes from connected components.")
        
        if not regions:
            st.warning("No regions detected. Adjust the threshold.")
        else:
            data_list = []
            for i, prop in enumerate(regions):
                data_list.append({
                    "Object": i + 1,
                    "Area": prop.area,
                    "Perimeter": round(prop.perimeter, 2),
                    "Compactness": round((prop.perimeter**2) / (4 * np.pi * prop.area), 2) if prop.area > 0 else 0,
                    "Eccentricity": round(prop.eccentricity, 2),
                    "Solidity": round(prop.solidity, 2)
                })
            
            st.table(data_list)
            st.info("**Compactness**: 1.0 is a perfect circle. Higher values mean more irregular/elongated shapes.")
            st.info("**Eccentricity**: 0 is a circle, values near 1 are very elongated.")

    with tab_topo:
        st.subheader("Topological Descriptors")
        st.markdown(r"Euler Number ($E$) = Connected Components ($C$) - Holes ($H$)")
        
        if regions:
            selected_obj = st.selectbox("Select Object to Analyze", range(1, len(regions)+1))
            prop = regions[selected_obj - 1]
            
            # Crop to the selected object
            minr, minc, maxr, maxc = prop.bbox
            obj_crop = binary[minr:maxr, minc:maxc]
            
            col1, col2 = st.columns(2)
            col1.image(obj_crop.astype(np.uint8)*255, caption=f"Object {selected_obj}", use_container_width=True)
            
            euler = prop.euler_number
            # Simple hole detection logic for visualization
            num_holes = 1 - euler if euler <= 1 else 0 # Simplified for single objects
            
            with col2:
                st.metric("Euler Number (E)", euler)
                st.write(f"This indicates the object has {num_holes} internal holes.")
                st.caption("Topological features remain unchanged by rotation, scaling, or stretching.")

    with tab_texture:
        st.subheader("Statistical Texture Descriptors")
        st.markdown("Measuring the 'look and feel' of intensity distributions.")
        
        # Calculate global stats
        mean_val = np.mean(img)
        std_val = np.std(img)
        
        # Calculate entropy (rough estimate from histogram)
        hist, _ = np.histogram(img, bins=256, range=(0, 255), density=True)
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        t1, t2, t3 = st.columns(3)
        t1.metric("Mean Intensity", round(mean_val, 2))
        t2.metric("Std Dev (Contrast)", round(std_val, 2))
        t3.metric("Entropy (Randomness)", round(entropy, 2))
        
        st.info("High Entropy = Complex texture/randomness. Low Entropy = Uniform region.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Boundary Processing**: Tracing and representing the outer shell.
    - **Chain Codes**: Compact direction-based encoding.
    - **Shape Numbers**: Normalized descriptors for scale/rotation invariance.
    - **Regional Descriptors**: Area, Perimeter, Compactness, Euler Number.
    - **Texture**: Statistical (Mean, Variance, Entropy) or Spectral (FFT).
    """)
