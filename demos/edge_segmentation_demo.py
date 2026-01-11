# demos/edge_segmentation_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, morphology, data, util, color, exposure
from PIL import Image
import io

@st.cache_data
def load_img_edge_seg(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Camera or Coins are good for edge-based segmentation
        img_np = data.camera()
    return img_np

def run():
    st.header("🧩 4.12 Edge-based Segmentation")
    st.markdown("""
    Edge-based segmentation follows the 'Cracks' in an image—sharp intensity changes that mark the boundaries of objects.
    """)

    with st.expander("📚 Theory: Finding the Cracks", expanded=False):
        st.markdown(r"""
        ### 1. Detection (The Gradient)
        We first find local intensity changes using derivative operators like **Sobel** or **Canny**. This creates an edge strength map.
        
        ### 2. Thresholding (The Binary Map)
        Weak gradient responses (noise) are discarded by applying a threshold. This produces a binary "edge skeleton."
        
        ### 3. Linking & Refinement
        Edge maps are often fragmented. We use **Morphological Operations** (like Dilation or Closing) to bridge gaps and create continuous contours.
        
        ### 🧩 Glued Jigsaw Puzzle Analogy
        Imagine a puzzle painted entirely green. You can't see the pieces by color, so you feel for the **cracks** (edges) between them to know where one piece ends and another begins.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="edge_seg_local")
    img = load_img_edge_seg(local_up.read() if local_up else None)
    img_float = util.img_as_float(img)

    st.sidebar.markdown("### 🧰 Pipeline Control")
    detector = st.sidebar.selectbox("Phase 1: Detector", ["Sobel", "Canny"])
    
    if detector == "Canny":
        sigma = st.sidebar.slider("Canny Sigma", 0.5, 5.0, 2.0)
        edges = feature.canny(img_float, sigma=sigma)
    else:
        # Sobel + Manual Thresholding
        edge_strength = filters.sobel(img_float)
        thresh = st.sidebar.slider("Phase 2: Strength Threshold", 0.01, 0.5, 0.1)
        edges = edge_strength > thresh

    refine_type = st.sidebar.selectbox("Phase 3: Refinement", ["None", "Dilation", "Closing"])
    radius = st.sidebar.slider("Refinement Radius", 1, 10, 2)
    
    if refine_type == "Dilation":
        final_seg = morphology.binary_dilation(edges, morphology.disk(radius))
    elif refine_type == "Closing":
        final_seg = morphology.binary_closing(edges, morphology.disk(radius))
    else:
        final_seg = edges

    tab_stages, tab_comparison, tab_table = st.tabs(["🏗️ Three-Stage Pipeline", "🆚 Method Comparison", "📊 Summary Profile"])

    with tab_stages:
        st.subheader("Interactive Segmentation Pipeline")
        
        c1, c2, c3 = st.columns(3)
        
        # Original/Phase 1
        if detector == "Sobel":
            c1.image(exposure.rescale_intensity(filters.sobel(img_float), out_range=(0,1)), caption="1. Edge Strength (Sobel)", use_container_width=True)
        else:
            c1.image(img_float, caption="1. Source Image", use_container_width=True)
            
        # Phase 2
        c2.image(util.img_as_ubyte(edges), caption="2. Binary Edge Map", use_container_width=True)
        
        # Phase 3
        c3.image(util.img_as_ubyte(final_seg), caption=f"3. Refinement ({refine_type})", use_container_width=True)
        
        st.info("💡 **Observation**: Notice how 'Closing' helps connect broken edge segments into continuous boundaries.")

    with tab_comparison:
        st.subheader("When to use Edge-based Segmentation?")
        
        st.table({
            "Feature": ["Edge-based", "Region-based (Unit 4)"],
            "Focus": ["Boundaries/Cracks", "Internal pixel similarity"],
            "Noise Sensitivity": ["Highly Sensitive", "Less Sensitive"],
            "Object Types": ["High contrast outlines", "Homogeneous regions"],
            "Broken Parts?": ["Fragmented; needs linking", "Complete handles but may bleed"]
        })

    with tab_table:
        st.subheader("Step-by-Step Logic")
        st.table({
            "Step": ["1. Edge Detection", "2. Thresholding", "3. Linking", "4. Result"],
            "Description": [
                "Compute gradient intensity map",
                "Convert strength map to binary bits",
                "Bridge gaps via morphology",
                "Obtain closed object boundaries"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Concept**: Finding object boundaries where intensity changes abruptly.
    - **Detectors**: Sobel (Gradient-based) and Canny (Optimal Edge Search).
    - **Gap Filling**: Essential because noise and lighting cause broken edges.
    - **Analogy**: Feeling for cracks in a mono-colored jigsaw puzzle.
    """)
