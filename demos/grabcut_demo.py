# demos/grabcut_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

@st.cache_data
def load_img_grabcut(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data))
        # Ensure RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_np = np.array(img)
    else:
        # Load a sample image if none provided
        # We need something with a clear object, like the astronaut or coffee
        from skimage import data
        img_np = data.astronaut()
    return img_np

def run():
    st.header("✂️ 4.19 GrabCut Algorithm")
    st.markdown("""
    GrabCut is an iterative segmentation method that combines **Graph Cuts** and **Gaussian Mixture Models (GMM)** to extract foreground objects using a simple bounding box or scribbles.
    """)

    with st.expander("📚 Theory: Smart Scissors", expanded=False):
        st.markdown(r"""
        ### 1. The Core Idea
        GrabCut treats segmentation as an optimization problem. It builds a graph where:
        - **Nodes**: Pixels.
        - **Edges**: Connect neighbors (Spatial) and connect pixels to Source/Sink (Data).
        - **Weights**: Based on color similarity and edge strength.
        
        ### 2. The Process (Iterative)
        1. **Init (`rect`)**: User draws a box. Everything outside is **Sure Background**. Inside is **Unknown**.
        2. **GMM Learning**: Algorithm learns color distributions (GMMs) for Foreground and Background from the initial regions.
        3. **Graph Cut**: Using the GMMs, it runs a **Min-Cut/Max-Flow** algorithm to find the best separation (lowest energy).
        4. **Repeat**: The new segmentation updates the GMMs, and the process repeats until convergence.
        
        ### 3. The Result
        The algorithm assigns every pixel to one of four classes:
        - 0: Sure Background
        - 1: Sure Foreground
        - 2: Probable Background
        - 3: Probable Foreground
        
        ### ✂️ Smart Scissors Analogy
        It's like cutting a photo from a magazine. You draw a rough box around the person. The "Smart Scissors" look at the colors inside vs outside the box to guess where the person ends and the background begins. If it makes a mistake, you can give it a "Scribble" hint to fix it.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="gc_local")
    img_rgb = load_img_grabcut(local_up.read() if local_up else None)
    
    # Resize for performance if too large
    max_dim = 600
    h, w = img_rgb.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
        h, w = img_rgb.shape[:2]

    st.sidebar.markdown("### 🧰 Bounding Box")
    # Interactive Rect
    pad = 20
    rect_x = st.sidebar.slider("Rect X", 0, w-10, pad)
    rect_y = st.sidebar.slider("Rect Y", 0, h-10, pad)
    rect_w = st.sidebar.slider("Rect Width", 10, w-rect_x, w - 2*pad)
    rect_h = st.sidebar.slider("Rect Height", 10, h-rect_y, h - 2*pad)
    
    iters = st.sidebar.slider("Iterations", 1, 10, 5)

    tab_lab, tab_vis, tab_table = st.tabs(["✂️ Interactive GrabCut", "👁️ Mask Visualization", "📊 Summary Profile"])

    with tab_lab:
        st.subheader("Foreground Extraction")
        st.markdown(f"Segmenting area inside box: **(x={rect_x}, y={rect_y}, w={rect_w}, h={rect_h})**")
        
        # Visualize Rect
        img_rect = img_rgb.copy()
        cv2.rectangle(img_rect, (rect_x, rect_y), (rect_x+rect_w, rect_y+rect_h), (255, 0, 0), 2)
        
        # Prepare GrabCut
        mask = np.zeros(img_rgb.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (rect_x, rect_y, rect_w, rect_h)
        
        # Run GrabCut
        try:
            cv2.grabCut(img_rgb, mask, rect, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            st.error(f"GrabCut Error (Likely empty rect): {e}")

        # Final Mask: 0 & 2 are background, 1 & 3 are foreground
        # We modify the mask so that all sure (1) and probable (3) FG are 1, others 0
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        
        # Apply mask
        img_fg = img_rgb * mask2[:, :, np.newaxis]

        col1, col2 = st.columns(2)
        
        col1.image(img_rect, caption="1. Input with Bounding Box", use_container_width=True)
        col2.image(img_fg, caption=f"2. Extracted Foreground ({iters} iters)", use_container_width=True)
        
        if np.sum(mask2) == 0:
            st.warning("⚠️ Result is empty! Try adjusting the Bounding Box to cover the object better.")
        else:
            st.success("Extraction Complete! The background has been removed.")

    with tab_vis:
        st.subheader("Under the Hood: The 4 Classes")
        st.markdown("GrabCut assigns one of 4 labels to every pixel.")
        
        # Visualize the raw GrabCut mask (0,1,2,3)
        # 0: BG (Black), 1: FG (White), 2: Prob. BG (Dark Grey), 3: Prob. FG (Light Grey)
        # We scale them for visibility: 0->0, 1->255, 2->85, 3->170
        vis_mask = np.zeros_like(mask, dtype=np.uint8)
        vis_mask[mask == 0] = 0   # Sure BG
        vis_mask[mask == 1] = 255 # Sure FG
        vis_mask[mask == 2] = 60  # Prob BG
        vis_mask[mask == 3] = 180 # Prob FG
        
        c_v1, c_v2 = st.columns(2)
        c_v1.image(vis_mask, caption="Raw Mask Labels", use_container_width=True)
        
        # Legend
        st.markdown("""
        - ⬛ **Black**: Sure Background (Outside Box)
        - ⬜ **White**: Sure Foreground (Learned)
        - 🌫️ **Grey tones**: Probable regions (Refined by Graph Cut)
        """)

    with tab_table:
        st.subheader("GrabCut Framework")
        st.table({
            "Component": ["Initialization", "GMM", "Graph Cut", "Iteration"],
            "Role": [
                "User hint (Box/Scribble)",
                "Color Distribution Model",
                "Energy Minimization",
                "Refinement Loop"
            ],
            "Image Impact": [
                "Sets initial Known BG/Unknown",
                "Learns 'Object-like' colors",
                "Finds optimal boundary",
                "Tightens the segmentation"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Method**: Semi-automatic (requires user input).
    - **Models**: Uses two Gaussian Mixture Models (One for FG, One for BG).
    - **Optimization**: Uses Graph Cuts (Min-Cut/Max-Flow) to minimize energy.
    - **Iterative**: Does not solve in one pass; improves models based on previous segmentation.
    """)
