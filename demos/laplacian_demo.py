# demos/laplacian_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve, gaussian_filter
import io

@st.cache_data
def load_img_lap(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic shapes
        img_np = np.zeros((512, 512), dtype=np.uint8)
        img_np[150:350, 150:350] = 200
        # Soften slightly to make zero-crossings wider/visible
        img_np = gaussian_filter(img_np, sigma=1.0)
        img = Image.fromarray(img_np)
    return np.array(img)

def plot_kernel(kernel, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(kernel, cmap='coolwarm', vmin=-8, vmax=8)
    for (i, j), z in np.ndenumerate(kernel):
        ax.text(j, i, f'{z}', ha='center', va='center', fontsize=14, weight='bold')
    ax.set_title(title)
    ax.axis('off')
    return fig

def run():
    st.header("✨ Laplacian in Image Processing")
    st.markdown("""
    The shortest path to edge detection and sharpening. The Laplacian relies on **2nd order derivatives** to check if a pixel is different from its neighbors.
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: The Math of Isotropy", expanded=False):
        st.markdown(r"""
        ### 1. Mathematical Foundation
        - **Continuous**: $\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$
        - **Discrete (Finite Difference)**: $\nabla^2 f \approx f(x+1) + f(x-1) + f(y+1) + f(y-1) - 4f(x,y)$
        
        ### 2. Digital Approximation (Kernels)
        We use convolution masks where coefficients sum to zero (No response in constant areas).
        
        ### 3. Characteristics
        - **Isotropic**: Rotation invariant (unlike gradients which have direction).
        - **Zero-Crossings**: The output flips sign (+ to -) exactly at the edge center.
        - **Noise Sensitive**: Amplifies high frequencies violently.
        
        ### 4. Analogy: The Crumpled Paper 📄
        Gradient is like feeling a slope with your hand. 
        **Laplacian** is like running a fingertip over crumpled paper—it catches sharply on the creases (edges) and points, ignoring the flat parts completely.
        """)

    # --- Global Input ---
    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="lap_local")
    img_gray = load_img_lap(local_up.read() if local_up else None)
    
    tab_kernels, tab_sharpen, tab_log, tab_freq = st.tabs(["🎛️ Kernel Explorer", "✏️ Sharpening", "🌊 Zero-Crossings & LoG", "🌐 Frequency Domain"])

    with tab_kernels:
        st.subheader("Digital Approximations")
        
        k_type = st.radio("Select Kernel Type", ["Standard (4-neighbor)", "Enhanced (8-neighbor)"], horizontal=True)
        
        if k_type == "Standard (4-neighbor)":
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            desc = "Checks Up, Down, Left, Right only."
        else:
            kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
            desc = "Checks diagonals too (Stronger response)."
            
        col_k1, col_k2 = st.columns([1, 2])
        with col_k1:
            st.pyplot(plot_kernel(kernel, "Convolution Mask"))
            st.caption(desc)
            
        with col_k2:
            # Apply convolution
            lap_res = convolve(img_gray.astype(float), kernel)
            # Normalize for display
            disp_lap = np.clip(np.abs(lap_res), 0, 255).astype(np.uint8)
            st.image(disp_lap, caption=f"Laplacian Response ({k_type})", use_container_width=True)
            st.info("Notice: Flat areas are black (0 response). Edges light up.")

    with tab_sharpen:
        st.subheader("Image Sharpening")
        st.markdown(r"Formula: $g(x,y) = f(x,y) - c[\nabla^2 f(x,y)]$ (Subtracting because center is negative)")
        
        c = st.slider("Sharpening Factor (c)", 0.0, 5.0, 1.0)
        
        lap_feat = convolve(img_gray.astype(float), kernel) # Use selected kernel from previous tab
        
        # Sharpening typically subtracts the laplacian if center is negative index
        # If center is -4, subtracting it adds 4*center to the original pixel relation
        sharpened = img_gray.astype(float) - c * lap_feat
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        s1, s2 = st.columns(2)
        s1.image(img_gray, caption="Original", use_container_width=True)
        s2.image(sharpened, caption=f"Sharpened (c={c})", use_container_width=True)

    with tab_log:
        st.subheader("Zero-Crossings & LoG")
        
        col_z1, col_z2 = st.columns(2)
        
        with col_z1:
            st.markdown("**Zero-Crossing Visualization**")
            # Create a simple edge profile
            prof = np.zeros(100)
            prof[50:] = 100
            # Smooth it
            prof_s = gaussian_filter(prof, sigma=2)
            d2 = np.gradient(np.gradient(prof_s))
            
            fig_z, ax_z = plt.subplots(figsize=(6, 3))
            ax_z.plot(prof_s, 'k--', label='Edge Profile')
            ax_z.plot(d2 * 10, 'r', label='2nd Derivative') # Scale for visibility
            ax_z.axhline(0, color='gray', alpha=0.5)
            ax_z.set_title("Zero Crossing at Edge Center")
            ax_z.legend()
            st.pyplot(fig_z)
            
        with col_z2:
            st.markdown("**Laplacian of Gaussian (LoG)**")
            sigma_log = st.slider("Smoothing Sigma", 0.5, 5.0, 2.0)
            
            noisy = img_gray + np.random.normal(0, 10, img_gray.shape)
            
            # 1. Smooth
            smooth = gaussian_filter(noisy, sigma=sigma_log)
            # 2. Lap
            log_out = convolve(smooth, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
            
            st.image(np.clip(np.abs(log_out)*5, 0, 255).astype(np.uint8), caption="LoG Response (Stable Edge Detection)", use_container_width=True)

    with tab_freq:
        st.subheader("Frequency Domain View")
        st.markdown(r"Transfer Function: $H(u,v) = -4\pi^2(u^2 + v^2)$")
        st.caption("It's a parabola opening downwards—amplifies high frequencies (large u, v) significantly.")
        
        # 3D Plot
        u = np.linspace(-1, 1, 50)
        v = np.linspace(-1, 1, 50)
        U, V = np.meshgrid(u, v)
        H = - (U**2 + V**2) # Simplified shape
        
        fig_3d = plt.figure(figsize=(8, 5))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        surf = ax_3d.plot_surface(U, V, -H, cmap='viridis') # Inverted to show the 'cup' shape of magnitude
        ax_3d.set_title("Filter Magnitude Response (High-Pass)")
        ax_3d.set_xlabel("u (Freq)")
        ax_3d.set_ylabel("v (Freq)")
        ax_3d.set_zlabel("|H|")
        st.pyplot(fig_3d)

    # Summary
    st.divider()
    st.markdown("### 📋 Laplacian Profile")
    st.table({
        "Aspect": ["Output Type", "Sensitivity", "Direction", "Use Case"],
        "Details": ["Scalar (No direction)", "High (Amplifies Noise)", "Isotropic (Rotation Invariant)", "Sharpening, LoG"]
    })
