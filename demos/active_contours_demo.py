# demos/active_contours_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color, segmentation, filters 
from skimage.filters import gaussian
from skimage.segmentation import active_contour, morphological_chan_vese, checkerboard_level_set
from PIL import Image
import io

@st.cache_data
def load_img_snake(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Astronaut head is classic for snake fitting
        img_np = color.rgb2gray(data.astronaut())
        # Crop to the head for better demo speed/visibility
        img_np = img_np[0:250, 50:300]
    return img_np

def circle_points(resolution, center, radius):
    """Generate points for a circle (Traditional Snake initialization)."""
    radians = np.linspace(0, 2*np.pi, resolution)
    r = center[1] + radius*np.sin(radians)
    c = center[0] + radius*np.cos(radians)
    return np.array([r, c]).T

def store_evolution_in_list(callback):
    """Refactoring helper for storing evolution if needed (not supported by all skimage versions easily via callback)."""
    pass

def run():
    st.header("🐍 4.17 Active Contours (Snakes)")
    st.markdown("""
    Active contours are dynamic curves that evolve to fit object boundaries by minimizing an energy function.
    """)

    with st.expander("📚 Theory: The Energy Minimization", expanded=False):
        st.markdown(r"""
        ### 1. The Rubber Band Model
        Think of the snake as an elastic **rubber band**. It wants to:
        - **Shrink/Smooth** (Internal Energy): Effectively 'tension' and 'stiffness'.
        - **Snap to Edges** (External Energy): Image gradients pull it towards boundaries.
        
        ### 2. The Energy Equation
        The snake minimizes $E = E_{internal} + E_{external}$.
        - **Internal**: $\alpha$ (continuity) + $\beta$ (curvature).
        - **External**: $-\|\nabla I\|^2$ (Gradient Magnitude).
        
        ### 3. Morphological Snakes (Modern Variant)
        Instead of solving complex PDEs, we can use **Morphological Operators** (Dilation/Erosion) to evolve the curve.
        - **Level Sets**: The curve is the zero-level set of a 3D surface.
        - **Chan-Vese (ACWE)**: Active Contours Without Edges. Uses region statistics (mean intensities) instead of gradients. Great for noisy images or weak boundaries.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="ac_local")
    img = load_img_snake(local_up.read() if local_up else None)
    img = img_as_float(img)

    tab_traditional, tab_morph, tab_table = st.tabs(["🐍 Traditional Snake", "🦠 Morphological ACWE", "📊 Summary Profile"])

    with tab_traditional:
        st.subheader("Traditional Active Contour")
        st.markdown("Fit a snake to edges using Gradient Descent.")

        c1, c2 = st.columns(2)
        with c1:
            alpha = st.slider("Alpha (Smoothness/Tension)", 0.001, 0.5, 0.015, format="%.3f")
            beta = st.slider("Beta (Curvature/Stiffness)", 0.1, 10.0, 1.0)
        with c2:
            gamma = st.slider("Gamma (Step Size)", 0.001, 0.1, 0.001, format="%.3f")
            w_edge = st.slider("Edge Attraction", 1, 20, 10)
        
        iterations = st.slider("Iterations", 50, 500, 200, step=50)
        
        # Init circle
        h, w = img.shape
        center = (w//2, h//2)
        radius = min(w, h)//3
        init = circle_points(200, center, radius)
        
        # Run fit
        # We assume light object on dark background or use Gaussian to smooth
        snake = active_contour(gaussian(img, 3), init, alpha=alpha, beta=beta, gamma=gamma, w_edge=w_edge, max_num_iter=iterations)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap='gray')
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3, label='Initial Snake')
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3, label='Evolved Snake')
        ax.legend(loc='upper right')
        ax.set_title("Active Contour Fitting")
        ax.axis('off')
        st.pyplot(fig)
        
        st.info("💡 **Tip**: If the snake collapses to a point, try decreasing Alpha/Beta. If it's too jagged, increase Alpha.")

    with tab_morph:
        st.subheader("Morphological Snake (Chan-Vese)")
        st.markdown("Active Contours **Without Edges**. Works on region statistics (mean intensity).")
        
        iter_morph = st.slider("Morph-Iter", 10, 100, 35)
        smooth = st.slider("Smoothness (Lambda)", 1, 5, 1) # Note: skimage implementation parameters vary, using 'lambda1/2' usually or 'iter' count of smoothing
        
        # Chan-Vese (ACWE) requires a level-set init
        init_ls = checkerboard_level_set(img.shape, 6)
        
        # Run Evolution
        # Using morphological_chan_vese as it's robust to noise
        ls = morphological_chan_vese(img, num_iter=iter_morph, init_level_set=init_ls, smoothing=smooth)
        
        c_m1, c_m2 = st.columns(2)
        c_m1.image(img, caption="Original Image", use_container_width=True)
        
        fig_m, ax_m = plt.subplots()
        ax_m.imshow(img, cmap='gray')
        # Contour the level set (0.5 is the boundary for binary mask)
        ax_m.contour(ls, [0.5], colors='r', linewidths=2)
        ax_m.set_title(f"ACWE Segmentation ({iter_morph} iters)")
        ax_m.axis('off')
        c_m2.pyplot(fig_m)
        
        st.success("Notice how this method finds objects (like eyes/skin) based on average brightness, even if edges are weak or missing.")

    with tab_table:
        st.subheader("Comparison: Energy vs. Morphology")
        st.table({
            "Feature": ["Traditional Snake", "Morphological ACWE"],
            "Energy Source": ["Edges (Gradient)", "Region Stats (Mean)"],
            "Initialization": ["Manual Curve (near object)", "Level Set (Grid/Checkers)"],
            "Topology": ["Fixed (cannot split)", "Flexible (can split/merge)"],
            "Best For": ["Tracing clean object boundaries", "Cluttered/Noisy blobs"]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Internal Energy**: Controls curve shape (Smoothness/Stiffness).
    - **External Energy**: Pulls curve to image features (Edges).
    - **Minimization**: Iteratively moves points to lower total energy.
    - **Chan-Vese**: A powerful variant that uses region homogeneity instead of gradients, allowing it to segment objects without defined edges.
    """)
