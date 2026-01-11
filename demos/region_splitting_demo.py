# demos/region_splitting_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import data, util, color
from PIL import Image
import io

@st.cache_data
def load_img_splitting(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Camera is great for showing blocky structures
        img_np = data.camera()
    return img_np

def split_recursive(img, x, y, w, h, threshold, min_size, regions):
    """
    Recursively splits the image into quadrants if variance > threshold.
    img: 2D numpy array
    x, y: Top-left corner
    w, h: Width and height
    threshold: Standard deviation limit for splitting
    min_size: Minimum block size to stop splitting
    regions: List to store (x, y, w, h, mean_intensity)
    """
    region = img[y:y+h, x:x+w]
    if region.size == 0:
        return

    std_dev = np.std(region)
    mean_val = np.mean(region)

    # Predicate Q(R): If std_dev > threshold and big enough, split
    if std_dev > threshold and w > min_size and h > min_size:
        half_w = w // 2
        half_h = h // 2
        # Quadtree Split
        split_recursive(img, x, y, half_w, half_h, threshold, min_size, regions) # Q1
        split_recursive(img, x + half_w, y, w - half_w, half_h, threshold, min_size, regions) # Q2
        split_recursive(img, x, y + half_h, half_w, h - half_h, threshold, min_size, regions) # Q3
        split_recursive(img, x + half_w, y + half_h, w - half_w, h - half_h, threshold, min_size, regions) # Q4
    else:
        # Homogeneous or too small
        regions.append((x, y, w, h, mean_val))

def run():
    st.header("📂 4.14 Region Splitting (Top-down)")
    st.markdown("""
    Region splitting is a top-down approach that starts with the whole image and recursively divides it into smaller quadrants until every part is 'uniform enough'.
    """)

    with st.expander("📚 Theory: Recursive Quadtrees", expanded=False):
        st.markdown(r"""
        ### 1. The Strategy: Top-Down
        Unlike region growing (bottom-up), splitting starts with the **entire image** as one big region. If it's too complex, we split it into 4 child regions (quadrants).
        
        ### 2. The Homogeneity Predicate $Q(R)$
        We use a rule to check if a region $R$ is uniform. A common rule is based on **Standard Deviation** ($\sigma_R$):
        - **If $\sigma_R > T$**: Region is too "noisy" or has multiple objects $\to$ **SPLIT**.
        - **If $\sigma_R \le T$**: Region is homogeneous $\to$ **STOP**.
        
        ### 3. The Quadtree Structure
        This recursive splitting naturally forms a **Quadtree** data structure, where the root is the image and every node has up to 4 children.
        
        ### 4. The Need for Merging
        Splitting alone often creates "artificial" boundaries because of the rigid grid. A **Merge** step usually follows to combine adjacent quadrants that are actually similar.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="rs_local")
    img = load_img_splitting(local_up.read() if local_up else None)
    img_float = util.img_as_float(img)
    h, w = img.shape

    st.sidebar.markdown("### 🧰 Splitting Controls")
    threshold = st.sidebar.slider("Homogeneity Threshold (Std Dev)", 0.01, 0.50, 0.15)
    min_size = st.sidebar.slider("Minimum Region Size", 4, 32, 8)
    
    tab_lab, tab_logic, tab_table = st.tabs(["📂 Quadtree Lab", "🧠 Procedure Logic", "📊 Summary Profile"])

    with tab_lab:
        st.subheader("Interactive Quadtree Decomposition")
        st.markdown(f"Splitting regions where $\sigma_R > {threshold:.2f}$.")
        
        # Run recursive split
        regions = []
        split_recursive(img_float, 0, 0, w, h, threshold, min_size, regions)
        
        # Visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Image with grid overlay
        ax[0].imshow(img, cmap='gray')
        for x, y, rw, rh, m in regions:
            rect = patches.Rectangle((x, y), rw, rh, linewidth=0.5, edgecolor='yellow', facecolor='none')
            ax[0].add_patch(rect)
        ax[0].set_title("Recursive Splitting Grid")
        ax[0].axis('off')
        
        # 2. Reconstructed "Mean" Image
        recon = np.zeros_like(img_float)
        for x, y, rw, rh, m in regions:
            recon[y:y+rh, x:x+rw] = m
        ax[1].imshow(recon, cmap='gray')
        ax[1].set_title("Homogeneous Regions (Means)")
        ax[1].axis('off')
        
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        col1.metric("Total Regions", len(regions))
        col2.info("💡 **Observation**: Areas with high detail (camera tripod, hair) are split into tiny blocks, while smooth backgrounds remain as large blocks.")

    with tab_logic:
        st.subheader("The Splitting Procedure")
        st.markdown("""
        1. **Start**: Define the whole image as the initial region $R$.
        2. **Test**: Calculate standard deviation $\sigma_R$.
        3. **Decision**: Is $\sigma_R > T$?
           - **Yes**: Subdivide into 4 quadrants and repeat for each.
           - **No**: Stop splitting for this quadrant.
        4. **Merge (Optional)**: After splitting is done, check if neighbors are similar enough to be combined.
        """)
        
        st.info("📂 **Quadtree Analogy**: Think of a file system folders. If a folder gets too 'full' of different things, you create 4 sub-folders to organize them better.")

    with tab_table:
        st.subheader("Region Splitting Framework")
        st.table({
            "Component": ["Top-down", "Quadtree", "Predicate Q(R)", "Merge Step"],
            "Meaning": [
                "Start big → split smaller",
                "Each split gives 4 quadrants",
                "Rule to test homogeneity (variance/mean)",
                "Handles rigid quadrant structure artifacts"
            ],
            "Effect": [
                "Global perspective",
                "Hierarchical organization",
                "Decision marker for detail",
                "Connects separated similar blocks"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Split Rule**: Common to split if Variance ($\sigma^2$) or Std Dev ($\sigma$) exceeds a threshold.
    - **Structure**: Always recursive; naturally maps to a tree structure.
    - **Artifacts**: Can produce blocky boundaries ('staircase' effect) without a merge step.
    - **Best For**: Cluttered scenes where you want to adaptively focus detail on complex parts.
    """)
