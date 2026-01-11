# demos/segmentation_intro_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, color, filters, segmentation, data, util, exposure, transform
from skimage.morphology import disk
from scipy import ndimage as ndi
from PIL import Image
import io

@st.cache_data
def load_img_seg(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Coins are perfect for watershed/segmentation demo
        img_np = data.coins()
    return img_np

def run():
    st.header("🏙️ Image Segmentation: Partitioning the Visual World")
    st.markdown("""
    Image segmentation partitions an image into meaningful regions or objects. It transforms pixels into 'things'.
    """)

    with st.expander("📚 Theory: The Highlighter Analogy", expanded=False):
        st.markdown(r"""
        ### 1. Principal Types
        - **Semantic Segmentation**: Label every pixel with a class (e.g., all pixels that are "car").
        - **Instance Segmentation**: Label Every pixel AND separate individual objects (e.g., "Car 1", "Car 2").
        - **Boundary Detection**: Find the edges that separate regions.
        
        ### 2. Key Techniques
        - **Thresholding (Otsu)**: Finding the best intensity cutoff to separate foreground from background.
        - **Hough Transform**: Detecting geometric shapes (lines, circles) through a voting scheme.
        - **Watershed**: Treating image intensity as topography and "flooding" it to find regions.
        
        ### 🏙️ The City Map Analogy
         think of a cluttered city map. Segmentation is like using different colored highlighters to color parks, roads, and buildings so they are clearly separated.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="seg_local")
    img = load_img_seg(local_up.read() if local_up else None)
    img_float = util.img_as_float(img)

    tab_hough, tab_otsu, tab_watershed, tab_table = st.tabs([
        "📏 Hough Transform", "🌡️ Otsu Thresholding", "🌊 Watershed", "📊 Comparison Matrix"
    ])

    with tab_hough:
        st.subheader("Hough Transform: Line Detection")
        st.markdown("Transforming pixels into geometric shapes via parameter space voting.")
        
        edges = feature.canny(img_float, sigma=2.0)
        # Probabilistic Hough
        h, theta, d = transform.hough_line(edges)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(edges, cmap='gray')
        ax[0].set_title("Canny Edges")
        ax[1].imshow(img, cmap='gray')
        
        # Overlay lines
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
        _, angles, dists = transform.hough_line_peaks(h, theta, d, num_peaks=10)
        
        for angle, dist in zip(angles, dists):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            ax[1].axline((x0, y0), slope=np.tan(angle + np.pi/2), color='red', linewidth=1)
            
        ax[1].set_title("Detected Lines (Top Peaks)")
        for a in ax: a.axis('off')
        st.pyplot(fig)
        st.info("The Hough transform maps each edge pixel to a curve in $(\theta, \rho)$ space. Peaks in that space indicate lines.")

    with tab_otsu:
        st.subheader("Otsu's Optimal Thresholding")
        st.markdown("Automatically finding the 'valley' in a bimodal histogram.")
        
        val = filters.threshold_otsu(img)
        binary = img > val
        
        col1, col2 = st.columns(2)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(img.ravel(), bins=256, color='gray')
        ax_hist.axvline(val, color='red', linestyle='--', label=f'Otsu Thresh: {int(val)}')
        ax_hist.legend()
        ax_hist.set_title("Intensity Histogram")
        col1.pyplot(fig_hist)
        
        col2.image(binary, caption=f"Binary Result (Thresh={int(val)})", use_container_width=True)
        st.success("Otsu's method maximizes the variance between the two classes (foreground and background).")

    with tab_watershed:
        st.subheader("Watershed Algorithm: Separating Objects")
        st.markdown("Useful for separating touching objects by treating distance as elevation.")
        
        # 1. Denoise and threshold
        denoised = filters.rank.median(img, disk(2))
        markers = filters.rank.gradient(denoised, disk(5)) < 10
        markers = ndi.label(markers)[0]
        
        # 2. Distance transform
        distance = ndi.distance_transform_edt(img > val)
        # 3. Watershed
        labels = segmentation.watershed(-distance, markers, mask=img > val)
        
        fig_ws, ax_ws = plt.subplots(1, 2, figsize=(10, 5))
        ax_ws[0].imshow(-distance, cmap='viridis')
        ax_ws[0].set_title("Inverse Distance Map (Topography)")
        
        ax_ws[1].imshow(color.label2rgb(labels, image=img, bg_label=0))
        ax_ws[1].set_title("Watershed Segments")
        for a in ax_ws: a.axis('off')
        st.pyplot(fig_ws)
        st.warning("Watershed simulates 'flooding' from the centers. Lines form where 'catchment basins' meet.")

    with tab_table:
        st.subheader("Segmentation Method Summary")
        st.table({
            "Method": ["Semantic", "Instance", "Thresholding", "Watershed", "Hough"],
            "Category": ["Whole Image", "Object-wise", "Boundary-based", "Region-based", "Geometry-based"],
            "Best For": ["General scene labeling", "Counting objects", "High-contrast simple backgrounds", "Touching objects", "Lines/Circles"],
            "Limitation": ["No instance separation", "Expensive computations", "Sensitive to lighting", "Over-segmentation risk", "Fixed shapes only"]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Segmentation**: Partitioning image into meaningful regions.
    - **Otsu**: Optimization technique for global thresholding.
    - **Hough**: Parametric voting for shape detection.
    - **Watershed**: Topographical isolation of objects.
    - **Objective**: Simplify the image for high-level recognition.
    """)
