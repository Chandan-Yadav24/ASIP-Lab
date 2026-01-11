# demos/images_as_signals_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from PIL import Image

def load_image(uploaded_file):
    """Load and process image into numpy arrays."""
    img = Image.open(uploaded_file)
    # Convert to RGB for consistency
    img_rgb = img.convert('RGB')
    # Pre-scale for 3D and Matrix views to avoid performance hits
    img_small = img_rgb.resize((128, 128)) 
    img_gray = img_rgb.convert('L')
    img_gray_small = img_gray.resize((64, 64))
    
    return np.array(img_rgb), np.array(img_gray), np.array(img_gray_small), np.array(img_small)

def generate_test_image(size=128):
    """Generate a combined pattern if no image is uploaded."""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    img = 0.5 * X + 0.3 * Y
    square_mask = (X > 0.4) & (X < 0.6) & (Y > 0.4) & (Y < 0.6)
    img[square_mask] = 1.0
    return img

def run():
    st.subheader("IMAGE AS SIGNAL: Representing Visual Data")

    # --- Theory Section (Infographic Style) ----------------------------------
    with st.expander("📚 Core Concepts Overview", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            ### 1. Spatial Domain
            *   **Intensity & Amplitude**: $f(x, y)$ = pixel value at coord $(x, y)$.
            *   **Matrix Structure**: Discrete grid of values.
            ### 2. Frequency Domain
            *   **Spectral Decomposition**: Fourier Transform converts spatial to frequency.
            *   **Meaning**: Background = Low Freq; Edges = High Freq.
            """)
        with c2:
            st.markdown("""
            ### 3. Vector Representation
            *   **Multi-channel**: Each pixel is a vector (r, g, b tuple).
            *   **3D Array**: $H \times W \times 3$ for color images.
            ### 4. Analogy: Topographic Map
            *   **Coordinates**: $(x, y)$ location.
            *   **Elevation**: Signal value (intensity) = Height.
            """)

    st.markdown("---")

    # --- Input Section -------------------------------------------------------
    st.write("### 📥 Input Layer")
    uploaded_file = st.file_uploader("Upload your own image to analyze", type=['png', 'jpg', 'jpeg'], key="img_sig_upload")
    
    if uploaded_file:
        img_rgb, img_gray, img_gray_small, img_rgb_small = load_image(uploaded_file)
        source_label = "User Image"
    else:
        # Fallback to test pattern
        img_gray = generate_test_image(128)
        img_gray_small = generate_test_image(64)
        # Create a dummy RGB from grayscale
        img_rgb = np.stack([img_gray]*3, axis=-1)
        img_rgb_small = np.stack([img_gray_small]*3, axis=-1)
        source_label = "Default Pattern"

    # --- Dashboard Features --------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Spatial Grid", 
        "2. Frequency Spectrum", 
        "3. Vector (RGB)", 
        "4. Topo Analogy (3D)"
    ])

    # 1. Spatial Domain
    with tab1:
        st.write("### Spatial Domain: The Pixel Matrix")
        col_s1, col_s2 = st.columns([1, 1.5])
        with col_s1:
            st.image(img_rgb, caption=f"Original ({source_label})", use_container_width=True)
        with col_s2:
            st.markdown("**Local Matrix Patch (Top-Left 8x8)**")
            patch = (img_gray[:8, :8] * 255).astype(int) if not uploaded_file else img_gray[:8, :8]
            st.dataframe(patch)
            st.caption("Each cell represents the light Intensity at $(x, y)$.")

    # 2. Frequency Domain
    with tab2:
        st.write("### Frequency Domain: Spectral Decomposition")
        f_transform = fftshift(fft2(img_gray))
        magnitude = np.log(1 + np.abs(f_transform))
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Grayscale Signal**")
            fig_f1, ax_f1 = plt.subplots()
            ax_f1.imshow(img_gray, cmap='gray')
            ax_f1.axis('off')
            st.pyplot(fig_f1)
        with col_f2:
            st.markdown("**Fourier Transform (Magnitude)**")
            fig_f2, ax_f2 = plt.subplots()
            ax_f2.imshow(magnitude, cmap='turbo') # Use 'turbo' for infographic feel
            ax_f2.axis('off')
            st.pyplot(fig_f2)
        st.caption("Center = Low Freq (Base shapes). Outer regions = High Freq (Edges & Noise).")

    # 3. Vector Representation
    with tab3:
        st.write("### Vector Representation: RGB Channels")
        st.markdown(f"A color pixel is a vector $(r, g, b)$ in 3-dimensional color space.")
        
        fig_v, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ['R Channel', 'G Channel', 'B Channel']
        cmaps = ['Reds', 'Greens', 'Blues']
        
        for i in range(3):
            axes[i].imshow(img_rgb[:,:,i], cmap=cmaps[i])
            axes[i].set_title(titles[i])
            axes[i].axis('off')
        st.pyplot(fig_v)

    # 4. Topo Analogy
    with tab4:
        st.write("### Analogy: The Topographic Landscape")
        st.markdown("If pixel intensity is elevation, the image becomes a jagged terrain.")
        
        # Use a downsampled version for performance
        z = img_gray_small
        h, w = z.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        fig_3d = plt.figure(figsize=(10, 7))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        surf = ax_3d.plot_surface(x, y, z, cmap='viridis', edgecolor='none', alpha=0.9)
        ax_3d.set_title("Intensity Profile as Elevation Map")
        ax_3d.set_zlabel("Intensity")
        ax_3d.view_init(elev=50, azim=45) # Professional angle
        st.pyplot(fig_3d)

