# demos/region_growing_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util, color
from PIL import Image
import io
from collections import deque

@st.cache_data
def load_img_growing(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Photographer or coins image
        img_np = data.camera()
    return img_np

def region_growing(img, seed, threshold):
    """
    Implements basic region growing from a single seed point using BFS.
    img: 2D numpy array (grayscale)
    seed: tuple (x, y)
    threshold: intensity difference tolerance
    """
    height, width = img.shape
    segmented = np.zeros_like(img, dtype=bool)
    visited = np.zeros_like(img, dtype=bool)
    
    # Starting conditions
    seed_intensity = float(img[seed[1], seed[0]])
    queue = deque([seed])
    visited[seed[1], seed[0]] = True
    segmented[seed[1], seed[0]] = True
    
    # Neighborhood offsets (8-connectivity)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        cx, cy = queue.popleft()
        
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                if not visited[ny, nx]:
                    visited[ny, nx] = True
                    # Homogeneity Criterion: Compare with seed intensity
                    if abs(float(img[ny, nx]) - seed_intensity) <= threshold:
                        segmented[ny, nx] = True
                        queue.append((nx, ny))
    
    return segmented

def run():
    st.header("🌱 4.13 Region Growing Segmentation")
    st.markdown("""
    Region growing partitions an image by grouping similar pixels starting from initial 'seed' points.
    """)

    with st.expander("📚 Theory: From Seeds to Regions", expanded=False):
        st.markdown(r"""
        ### 1. The Strategy
        Unlike edge-detection (which finds boundaries), region growing finds the **inside** of objects by identifying clusters of similar pixels.
        
        ### 2. The Homogeneity Criterion
        A pixel is added to a region if its intensity is close to the region's properties:
        $$|I(x, y) - \mu_{Seed}| \le T$$
        - $I(x, y)$: Candidate pixel intensity.
        - $\mu_{Seed}$: Intensity of the starting point.
        - $T$: Acceptance threshold.
        
        ### 3. Iterative Expansion
        Starting from one pixel (**Seed**), the algorithm checks all neighbors (usually 8-connectivity). If they match the rule, they become part of the region and their neighbors are checked in turn.
        
        ### 🌿 Land Surveyor Analogy
        Imagine you're in a massive garden. You pick one piece of **red mulch** (the **Seed**). You tell your assistant: "Pick up every adjacent piece of material that is also red mulch." You stop when you hit green grass (the **Homogeneity Criterion** fails).
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="rg_local")
    img = load_img_growing(local_up.read() if local_up else None)
    h, w = img.shape

    st.sidebar.markdown("### 🧰 Growth Parameters")
    seed_x = st.sidebar.slider("Seed X Coordinate", 0, w-1, w//2)
    seed_y = st.sidebar.slider("Seed Y Coordinate", 0, h-1, h//2)
    threshold = st.sidebar.slider("Homogeneity Threshold (T)", 1, 100, 20)
    
    tab_lab, tab_props, tab_table = st.tabs(["🌱 Interactive Lab", "⚖️ Advantages & Limitations", "📊 Framework Summary"])

    with tab_lab:
        st.subheader("Seed-based Expansion")
        st.markdown(f"Starting seed at **({seed_x}, {seed_y})** with tolerance **{threshold}**.")
        
        # Run Region Growing
        mask = region_growing(img, (seed_x, seed_y), threshold)
        
        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original with seed marker
        ax[0].imshow(img, cmap='gray')
        ax[0].plot(seed_x, seed_y, 'ro', markersize=8, label='Seed Point')
        ax[0].set_title("Original & Seed Position")
        ax[0].axis('off')
        ax[0].legend()
        
        # Segmented Overlay
        # Color pixels based on the mask
        overlay = color.label2rgb(mask, image=img, colors=['red'], alpha=0.3, bg_label=0)
        ax[1].imshow(overlay)
        ax[1].set_title("Segmented Region (Red Overlay)")
        ax[1].axis('off')
        
        st.pyplot(fig)
        
        pixel_count = np.sum(mask)
        st.success(f"Region expanded to cover **{pixel_count:,}** pixels (**{(pixel_count/(w*h))*100:.1f}%** of image).")

    with tab_props:
        st.subheader("Method Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Advantages")
            st.success("""
            - **Efficient**: Simple logic, fast convergence.
            - **Irregular Shapes**: Great for organic boundaries.
            - **Multiple Seeds**: Can isolate many objects simultaneously.
            """)
            
        with col2:
            st.markdown("### ❌ Limitations")
            st.error("""
            - **Seed Dependent**: Result changes completely if seed is misplaced.
            - **Noise Sensitive**: A single noisy pixel can 'leak' the region into areas it shouldn't be.
            - **Shading Issues**: Fails if the object has a lighting gradient.
            """)

    with tab_table:
        st.subheader("Region Growing Framework")
        st.table({
            "Stage": ["Initialization", "Homogeneity", "Expansion", "Termination"],
            "Task": [
                "Select seed points manually or auto",
                "Define rule: |I - Seed| <= T",
                "Check 8-neighbors iteratively (BFS/DFS)",
                "No more neighbors satisfy the rule"
            ],
            "Goal": [
                "Find 'anchor' for the object",
                "Ensure regional similarity",
                "Grow the boundary outwards",
                "Finalize the segment area"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Approach**: Direct grouping (bottom-up), not boundary searching (top-down).
    - **Connectivity**: Uses 4- or 8-connectivity to check neighbors.
    - **Medical Utility**: Widely used to isolate tumors/organs in MRI/CT scans.
    - **Crucial Choice**: The seed point and threshold $T$ are the most important parameters.
    """)
