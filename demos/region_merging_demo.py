# demos/region_merging_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util, color, segmentation, measure
from PIL import Image
import io

@st.cache_data
def load_img_merging(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Coins or Coffee are good for region grouping
        img_np = data.coins()
    return img_np

def get_initial_regions(img, n_segments):
    """Uses SLIC superpixels to create an initial over-segmentation (villages)."""
    # SLIC works on RGB usually, convert grayscale accordingly
    img_rgb = color.gray2rgb(img)
    labels = segmentation.slic(img_rgb, n_segments=n_segments, compactness=10, start_label=1)
    return labels

def merge_regions(img, labels, threshold):
    """
    Simulates region merging. 
    Adjacent regions are merged if their mean intensity difference is below threshold.
    """
    # Compute region properties
    props = measure.regionprops(labels, intensity_image=img)
    means = {p.label: p.mean_intensity for p in props}
    
    # Simple iterative merging based on adjacency
    # We use a disjoint-set style mapping
    parent = {l: l for l in means.keys()}
    
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Find boundaries
    edge_map = segmentation.find_boundaries(labels, mode='inner')
    
    # Check neighbors across boundaries
    # This is a simplification: we'll check horizontal and vertical neighbors
    h, w = labels.shape
    for y in range(h):
        for x in range(w-1):
            l1, l2 = labels[y, x], labels[y, x+1]
            if l1 != l2:
                if abs(means[l1] - means[l2]) < threshold:
                    union(l1, l2)
                    
    for y in range(h-1):
        for x in range(w):
            l1, l2 = labels[y, x], labels[y+1, x]
            if l1 != l2:
                if abs(means[l1] - means[l2]) < threshold:
                    union(l1, l2)

    # Reconstruct new label map
    new_labels = np.zeros_like(labels)
    for l in means.keys():
        new_labels[labels == l] = find(l)
        
    return new_labels

def run():
    st.header("🤝 4.15 Region Merging (Refinement)")
    st.markdown("""
    Region merging is a bottom-up technique that combines smaller parts into larger, homogeneous 'provinces' based on similarity.
    """)

    with st.expander("📚 Theory: Villages into Provinces", expanded=False):
        st.markdown(r"""
        ### 1. The Strategy: Bottom-Up
        Merging is often the second step of **Split-and-Merge**. While splitting (top-down) finds detail, it often separates areas that are actually similar. Merging fixes this by looking at adjacent regions.
        
        ### 2. The Similarity Rule (Predicate)
        We merge two adjacent regions $R_1$ and $R_2$ if they are "similar enough." A classic rule is the **Mean Intensity Difference**:
        $$|\mu_1 - \mu_2| < T_{merge}$$
        - $\mu_1, \mu_2$: Average pixel intensity of the regions.
        - $T_{merge}$: The merge sensitivity.
        
        ### 3. Iterative Consolidation
        The process repeats: we keep merging neighboring groups until no more adjacent pairs satisfy the rule.
        
        ### 🏘️ Villages into Provinces Analogy
        Imagine many tiny **villages** (initial segments). If two neighboring villages speak the same language or have the same economy (**Similarity Rule**), they merge into a **Province**. You keep merging until every province is distinct from its neighbors.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="rm_local")
    img = load_img_merging(local_up.read() if local_up else None)
    img_float = util.img_as_float(img)
    
    st.sidebar.markdown("### 🧰 Merging Controls")
    n_init = st.sidebar.slider("Initial Villages (Over-segmentation)", 50, 500, 200)
    t_merge = st.sidebar.slider("Merge Threshold (T)", 0.01, 0.50, 0.10)
    
    tab_lab, tab_props, tab_table = st.tabs(["🤝 Merging Lab", "⚖️ Strengths & Risks", "📊 Summary Profile"])

    with tab_lab:
        st.subheader("Consolidating Regions")
        st.markdown(f"Starting with {n_init} villages. Merging if $| \mu_1 - \mu_2 | < {t_merge:.2f}$.")
        
        # Initial Over-segmentation
        labels_init = get_initial_regions(img_float, n_init)
        
        # Perform Merging
        labels_merged = merge_regions(img_float, labels_init, t_merge)
        
        c1, c2 = st.columns(2)
        
        # Visualize Initial
        fig1, ax1 = plt.subplots()
        ax1.imshow(segmentation.mark_boundaries(img_float, labels_init, mode='inner'))
        ax1.set_title(f"Initial Over-segmentation ({n_init} regions)")
        ax1.axis('off')
        c1.pyplot(fig1)
        
        # Visualize Merged
        fig2, ax2 = plt.subplots()
        ax2.imshow(segmentation.mark_boundaries(img_float, labels_merged, mode='inner'))
        ax2.set_title(f"After Merging ({len(np.unique(labels_merged))} regions)")
        ax2.axis('off')
        c2.pyplot(fig2)
        
        st.info("💡 **Observation**: Increasing **T** causes more regions to vanish into their neighbors, eventually leading to 'under-segmentation' where distinct objects are lost.")

    with tab_props:
        st.subheader("Method Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ✅ Advantages")
            st.success("""
            - **Flexible Shapes**: Not restricted to grids or cones.
            - **Clutter Handling**: Excellent for grouping noisy backgrounds.
            - **Refinement Power**: The perfect cleanup for blocky Quadtree splits.
            """)
            
        with col2:
            st.markdown("### ❌ Risks")
            st.error("""
            - **Under-segmentation**: High thresholds swallow objects.
            - **Over-segmentation**: Low thresholds leave 'salt' fragments.
            - **Heaviness**: Can be slow for high-res images with many tiny initial regions.
            """)

    with tab_table:
        st.subheader("Region Merging Framework")
        st.table({
            "Stage": ["Input", "Adjacency Check", "Decision", "Consolidation"],
            "Task": [
                "Over-segmented map (Splitting or Superpixels)",
                "Find neighbors sharing a boundary",
                "Apply rule: |Mean1 - Mean2| < T",
                "Re-label adjacent pixels to same ID"
            ],
            "Effect": [
                "Start with 'Villages'",
                "Check connection",
                "Decide if same 'Province'",
                "Iteratively grow regions"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Method Type**: Bottom-up (start small/fragmented $\to$ combine larger).
    - **Dependency**: Highly threshold-dependent ($T_{merge}$).
    - **Modern Spin**: Often uses **Region Adjacency Graphs (RAG)** to represent boundaries.
    - **Analogy**: Villages $\to$ Provinces $\to$ Countries.
    """)
