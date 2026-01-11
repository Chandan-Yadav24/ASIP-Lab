# demos/watershed_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util, color, filters, segmentation, morphology, feature
from scipy import ndimage as ndi
from PIL import Image
import io

@st.cache_data
def load_img_watershed(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Coins is the classic example for touching objects
        img_np = data.coins()
    return img_np

def run():
    st.header("🌊 4.16 Watershed Algorithm")
    st.markdown("""
    The watershed algorithm treats an image as a topographic map. It is particularly powerful for separating objects that are touching or overlapping.
    """)

    with st.expander("📚 Theory: Flooding the Valleys", expanded=False):
        st.markdown(r"""
        ### 1. Landscape Perspective
        - **Ridges (High Int.)**: Barriers or boundaries.
        - **Valleys (Low Int.)**: Catchment basins where water collects.
        
        ### 2. The Flooding Process
        1. **Gradient Compute**: Usually applied to the gradient magnitude so edges become tall mountains.
        2. **Minima/Markers**: Identify starting points (valleys).
        3. **Flooding**: Fill the landscape with water starting from the markers.
        4. **Dam Building**: When two basins meet, build a 'Dam' (watershed line) to keep them separate.
        
        ### 📏 Marker-Controlled Watershed
        Raw watershed often suffers from **Over-segmentation** (too many tiny regions from noise). We fix this by providing:
        - **Internal Markers**: Confirmed object pixels.
        - **External Markers**: Confirmed background pixels.
        
        ### 🌊 Flooding Analogy
        Imagine a village in a valley. If the valley floods, the water stays in that valley. When the water from your valley is about to spill into the next village's valley, a wall (Dam) is built. These walls eventually outline every distinct village.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="ws_local")
    img = load_img_watershed(local_up.read() if local_up else None)
    img_float = util.img_as_float(img)

    st.sidebar.markdown("### 🧰 Watershed Params")
    denoise = st.sidebar.slider("Denoise Strength (Median)", 1, 15, 3, step=2)
    marker_dist = st.sidebar.slider("Marker Sensitivity", 0.1, 1.0, 0.4)
    
    tab_lab, tab_topography, tab_table = st.tabs(["🌊 Separation Lab", "⛰️ Topographic View", "📊 Summary Profile"])

    with tab_lab:
        st.subheader("Separating Touching Objects")
        
        # Step 1: Denoise
        denoised = filters.median(img, morphology.disk(denoise))
        
        # Step 2: Threshold to find objects
        # We use Otsu as a baseline
        thresh = filters.threshold_otsu(denoised)
        bw = denoised > thresh
        
        # Step 3: Distance Transform
        # Find distance to background to find "peaks" of objects
        distance = ndi.distance_transform_edt(bw)
        
        # Step 4: Markers
        # Peaks in distance map are seeds
        coords = feature.peak_local_max(distance, min_distance=int(20*marker_dist), labels=bw)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        
        # Step 5: Watershed
        # Grow from markers onto the inverted distance map (so peaks are valleys)
        labels = segmentation.watershed(-distance, markers, mask=bw)
        
        c1, c2, c3 = st.columns(3)
        
        # Raw Threshold
        c1.image(util.img_as_ubyte(bw), caption="1. Simple Threshold (Touching)", use_container_width=True)
        
        # Distance Map
        c2.image(util.img_as_ubyte(distance / distance.max()), caption="2. Distance Map (Peaks)", use_container_width=True)
        
        # Final Watershed
        fig_w, ax_w = plt.subplots()
        ax_w.imshow(color.label2rgb(labels, image=img_float, bg_label=0))
        ax_w.axis('off')
        c3.pyplot(fig_w)
        c3.caption("3. Separated by Watershed")
        
        st.success(f"Successfully isolated **{len(np.unique(labels))-1}** distinct objects.")
        st.info("💡 **Insight**: Notice how the watershed found boundaries even where coins were physically touching.")

    with tab_topography:
        st.subheader("The 'Elevation' Concept")
        st.markdown("Watershed sees the image as a series of ridges and valleys.")
        
        # Gradient Magnitude
        gradient = filters.sobel(img_float)
        
        col_t1, col_t2 = st.columns(2)
        
        col_t1.image(util.img_as_ubyte(gradient / (gradient.max() or 1)), caption="Gradient Map (Boundaries = Ridges)", use_container_width=True)
        
        # 3D surface plot
        fig_3d = plt.figure(figsize=(10, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        # Downsample for performance
        ds = 4
        X, Y = np.meshgrid(np.arange(0, img.shape[1], ds), np.arange(0, img.shape[0], ds))
        Z = img[::ds, ::ds]
        ax_3d.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8)
        ax_3d.set_title("Intensity Topography")
        ax_3d.set_zlabel("Intensity")
        col_t2.pyplot(fig_3d)

    with tab_table:
        st.subheader("Watershed Algorithm Framework")
        st.table({
            "Element": ["Regional Minima", "Catchment Basin", "Watershed Line", "Markers"],
            "Meaning": ["Valley bottoms", "Area draining to one minimum", "Ridge separating basins", "Seed points for regions"],
            "Image Interpretation": ["Seed locations", "One Segmented Object", "Final Object Boundary", "Prevent Over-segmentation"]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Strength**: Unbeatable for separating **touching or overlapping** objects.
    - **Topography**: Elevation is typically **Gradient Magnitude** or **Inverse Distance**.
    - **Challenge**: Over-segmentation from noise (requires Marker-Control).
    - **Analogy**: Water flooding from different valleys until they meet.
    """)
