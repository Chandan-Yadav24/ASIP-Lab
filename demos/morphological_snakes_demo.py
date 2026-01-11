# demos/morphological_snakes_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color, segmentation, filters
from skimage.segmentation import morphological_chan_vese, morphological_geodesic_active_contour, checkerboard_level_set, inverse_gaussian_gradient
from PIL import Image
import io

@st.cache_data
def load_img_morph_snake(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Coins/Moon for ACWE, Camera/Astronaut for GAC
        img_np = data.coins()
    return img_np

def store_evolution_in_list(callback):
    """Helper to store evolution via callback."""
    evolution = []
    def _callback(level_set):
        evolution.append(level_set)
    return evolution, _callback

def circle_level_set(shape, center, radius):
    """Generates a binary level set for a circle (Polyfill for older skimage)."""
    r, c = np.mgrid[:shape[0], :shape[1]]
    dist = np.sqrt((r - center[0])**2 + (c - center[1])**2)
    return dist <= radius

def run():
    st.header("🧬 4.18 Morphological Snakes")
    st.markdown("""
    Morphological Snakes maximize efficiency and stability by using morphological operations (dilation/erosion) to evolve the contour, rather than solving complex PDEs.
    """)

    with st.expander("📚 Theory: Vacuum-Sealing the Image", expanded=False):
        st.markdown(r"""
        ### 1. Why Morphological?
        - **Faster**: Updates are simple binary operations.
        - **Stable**: Less numerical instability than floating-point PDE solvers.
        - **Flexible**: Handles topology changes (splitting/merging) naturally.
        
        ### 2. Two Main Flavors
        1. **MorphACWE (Chan-Vese)**: **Active Contours Without Edges**. 
           - Uses **Region Statistics** (Mean Intensity).
           - Works on images with weak/missing edges but distinct regions.
        2. **MorphGAC (Geodesic)**: **Geodesic Active Contours**.
           - Uses **Image Gradients** (Edges).
           - Attracted to high-contrast boundaries.
           
        ### 3. The Energy Balance
        $$E = E_{smooth} + \alpha E_{attract} + \beta E_{morph}$$
        The snake balances staying smooth vs. being pulled by the image features.
        
        ### 🧥 Vacuum-Sealing Analogy
        Traditional snakes are like stretching a rubber band. Morphological snakes are like **vacuum-sealing** a bag around a sweater. The bag shrinks rapidly and conforms to every lump and bump (irregular shape) using simple pressure (morphology) rather than complex elastic physics.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="ms_local")
    img = load_img_morph_snake(local_up.read() if local_up else None)
    img_float = img_as_float(img)

    tab_acwe, tab_gac, tab_table = st.tabs(["🦠 MorphACWE (Region)", "🌋 MorphGAC (Edge)", "📊 Summary Profile"])

    with tab_acwe:
        st.subheader("Morphological ACWE (Chan-Vese)")
        st.markdown("**Best for**: Objects with uniform interiors, even if edges are fuzzy or noisy.")
        
        iter_acwe = st.slider("Iterations", 10, 100, 30, key="acwe_iter")
        smooth_acwe = st.slider("Smoothing", 1, 5, 1, key="acwe_smooth")
        init_type = st.radio("Initialization", ["Checkerboard", "Circle"], horizontal=True)
        
        if init_type == "Checkerboard":
            init_ls = checkerboard_level_set(img.shape, square_size=min(img.shape)//8)
        else:
            init_ls = circle_level_set(img.shape, center=(img.shape[0]//2, img.shape[1]//2), radius=min(img.shape)//3)
            
        # Run
        ls_acwe = morphological_chan_vese(img_float, num_iter=iter_acwe, init_level_set=init_ls, smoothing=smooth_acwe)
        
        c1, c2 = st.columns(2)
        
        fig1, ax1 = plt.subplots()
        ax1.imshow(img, cmap='gray')
        ax1.contour(init_ls, [0.5], colors='r', linestyles='--')
        ax1.set_title("Initial Level Set")
        ax1.axis('off')
        c1.pyplot(fig1)
        
        fig2, ax2 = plt.subplots()
        ax2.imshow(img, cmap='gray')
        ax2.contour(ls_acwe, [0.5], colors='g', linewidths=2)
        ax2.set_title(f"ACWE Result ({iter_acwe} iters)")
        ax2.axis('off')
        c2.pyplot(fig2)
        
        st.info("💡 **Observation**: ACWE aims for 'Global Uniformity'. It tries to separate the image into two regions (Inside/Outside) that have minimal intensity variance.")

    with tab_gac:
        st.subheader("Morphological GAC (Geodesic)")
        st.markdown("**Best for**: Objects with visible boundaries/contours. Uses an **Inverse Gaussian Gradient** to stop the snake at edges.")
        
        iter_gac = st.slider("Iterations", 10, 300, 100, key="gac_iter")
        smooth_gac = st.slider("Smoothing", 1, 5, 2, key="gac_smooth")
        threshold_gac = st.slider("Balloon Force (Threshold)", -1.0, 1.0, 0.5, help="Positive expands, Negative shrinks")
        
        # Preprocessing: Inverse Gaussian Gradient
        gimage = inverse_gaussian_gradient(img_float)
        
        # Init: usually a small circle in the middle to grow OUTWARDS (Balloon force > 0)
        # or large circle to shrink INWARDS (Balloon force < 0)
        init_ls_gac = circle_level_set(img.shape, center=(img.shape[0]//2, img.shape[1]//2), radius=min(img.shape)//4)
        
        # Run
        ls_gac = morphological_geodesic_active_contour(gimage, num_iter=iter_gac, init_level_set=init_ls_gac, smoothing=smooth_gac, balloon=threshold_gac)
        
        c_g1, c_g2, c_g3 = st.columns(3)
        
        c_g1.image(img_float, caption="Original", use_container_width=True)
        c_g2.image(gimage, caption="Inverse Gradient (Attractor)", use_container_width=True)
        
        fig_g, ax_g = plt.subplots()
        ax_g.imshow(img, cmap='gray')
        ax_g.contour(ls_gac, [0.5], colors='cyan', linewidths=2)
        ax_g.set_title(f"GAC Result")
        ax_g.axis('off')
        c_g3.pyplot(fig_g)
        
        st.success("The **Inverse Gradient** is bright in flat areas (allowing movement) and dark at edges (stopping the snake).")

    with tab_table:
        st.subheader("Morphological Snake Framework")
        st.table({
            "Method": ["MorphACWE (Chan-Vese)", "MorphGAC (Geodesic)"],
            "Driver": ["Region Statistics (Mean)", "Edge Strength (Gradient)"],
            "Edge Requirement": ["None (Works on weak edges)", "High (Needs visible boundary)"],
            "Analogy": ["Separating Milk from Coffee", "Water stoppping at a Dam"],
            "Best For": ["Noisy, textured blobs", "Sharp-edged objects"]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Efficiency**: Morphological snakes are significantly faster/stable because they use binary morphology instead of PDEs.
    - **ACWE**: Minimizes variance inside/outside regions. Region-based.
    - **GAC**: Minimizes length along object boundaries. Edge-based.
    - **Balloon Force**: In GAC, used to inflate (expand) or deflate (shrink) the contour towards edges.
    """)
