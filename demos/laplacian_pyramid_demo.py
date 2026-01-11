# demos/laplacian_pyramid_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
import io

@st.cache_data
def load_img_laplacian(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Synthetic detailed image
        img_np = np.zeros((512, 512), dtype=np.uint8)
        # Add some shapes and patterns
        for i in range(0, 512, 64):
            img_np[i:i+2, :] = 255
            img_np[:, i:i+2] = 255
        y, x = np.ogrid[:512, :512]
        mask = (x - 256)**2 + (y - 256)**2 <= 100**2
        img_np[mask] = 180
        img = Image.fromarray(img_np)
    return np.array(img).astype(float)

def build_pyramids(img, levels, sigma):
    """Build both Gaussian and Laplacian pyramids."""
    # Gaussian Pyramid
    gaussian_pyramid = [img]
    current = img
    for _ in range(levels - 1):
        # Blur + Subsample
        blurred = gaussian_filter(current, sigma=sigma)
        current = blurred[::2, ::2]
        gaussian_pyramid.append(current)
    
    # Laplacian Pyramid
    laplacian_pyramid = []
    for i in range(levels - 1):
        g_i = gaussian_pyramid[i]
        g_next = gaussian_pyramid[i + 1]
        
        # 1. Upsample g_next back to size of g_i
        # Using simple nearest or bilinear-like interpolation
        # In scipy we can use zoom, but a simple Kronecker/Indexing expansion works for 2x
        # For better results, we normally use a 'pyrUp' like process (Upsample + Interpolation)
        from scipy.ndimage import zoom
        upsampled = zoom(g_next, 2, order=1) # Bilinear interpolation
        
        # Adjust size if it's off by 1 due to odd shapes
        if upsampled.shape != g_i.shape:
            upsampled = upsampled[:g_i.shape[0], :g_i.shape[1]]
            
        # 2. Subtract
        detail = g_i - upsampled
        laplacian_pyramid.append(detail)
        
    # The last element is the smallest Gaussian core
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    return gaussian_pyramid, laplacian_pyramid

def reconstruct_image(laplacian_pyramid):
    """Reconstruct original image from Laplacian levels."""
    current = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        from scipy.ndimage import zoom
        upsampled = zoom(current, 2, order=1)
        # Size match
        l_i = laplacian_pyramid[i]
        if upsampled.shape != l_i.shape:
            upsampled = upsampled[:l_i.shape[0], :l_i.shape[1]]
        current = upsampled + l_i
    return current

def run():
    st.header("🧱 Laplacian Pyramid: Capturing the Missing Details")
    st.markdown("""
    The **Laplacian Pyramid** stores what the Gaussian pyramid "throws away" at each level. 
    It's a collection of high-frequency detail images that, when added back to a blurry base, reconstruct the original perfectly.
    """)

    with st.expander("📚 Theory: What's Missing Between Levels?", expanded=False):
        st.markdown(r"""
        ### 1. Intuition
        - **Gaussian Pyramid**: "The Summary" (what remains after blurring and shrinking).
        - **Laplacian Pyramid**: "The Details" (what was lost during that summary step).
        
        ### 2. Construction Mechanism
        For each level $i$:
        $$L_i = G_i - \text{Upsample}(G_{i+1})$$
        - $G_i$: Current level.
        - $G_{i+1}$: Coarser (blurry, smaller) level.
        - $L_i$: The "Laplacian" (detail map).
        
        ### 3. Reconstruction
        Rebuilding the image is the reverse:
        1. Start with the smallest Gaussian core $G_n$.
        2. Upsample it and add the Laplacian detail $L_{n-1}$.
        3. Repeat until you reach Level 0.
        
        ### 4. Why Use It?
        - **Compression**: Laplacian images are mostly mid-gray (near zero); easy to compress.
        - **Blending**: Merge images scale-by-scale for seamless results.
        - **Sharpening**: Boosting specific Laplacian levels enhances fine vs coarse textures.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="lap_pyr_local")
    img_gray = load_img_laplacian(local_up.read() if local_up else None)
    
    levels = st.slider("Pyramid Levels", 2, 6, 4)
    sigma = st.slider("Smoothing Sigma", 0.5, 3.0, 1.0)
    
    gp, lp = build_pyramids(img_gray, levels, sigma)

    tab_visual, tab_reconstruct, tab_details = st.tabs(["🏗️ Build & Visualize", "🧩 Reconstruction Lab", "🧐 Detail Inspector"])

    with tab_visual:
        st.subheader("The Laplacian Stack")
        st.markdown("Each $L_i$ image shows detail at that specific scale. 0 is finest, top is coarsest.")
        
        fig, axes = plt.subplots(1, levels, figsize=(12, 3))
        for i in range(levels):
            # The last one is a Gaussian, others are Laplacian differences
            if i < levels - 1:
                # Difference images are centered at 0, shift for display
                disp = lp[i].copy()
                disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8) * 255
                axes[i].imshow(disp, cmap='gray')
                axes[i].set_title(f"Detail L_{i}")
            else:
                axes[i].imshow(lp[i], cmap='gray')
                axes[i].set_title(f"Core G_{i}")
            axes[i].axis('off')
        st.pyplot(fig)
        st.info("Laplacian levels look like fine gray outlines - these are the 'high frequencies'.")

    with tab_reconstruct:
        st.subheader("Perfect Reconstruction Test")
        st.markdown("If we add the details back correctly, we get the **exact** original image.")
        
        reconstructed = reconstruct_image(lp)
        diff = np.abs(img_gray - reconstructed)
        
        c1, c2, c3 = st.columns(3)
        c1.image(img_gray.astype(np.uint8), caption="Original (G_0)", use_container_width=True)
        c2.image(reconstructed.astype(np.uint8), caption="Reconstructed", use_container_width=True)
        
        # Visualize error (should be near black)
        err_disp = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
        c3.image(err_disp, caption="Error (Original - Recon)", use_container_width=True)
        
        max_err = np.max(diff)
        st.write(f"**Max Numerical Error:** {max_err:.4e}")
        if max_err < 1e-5:
            st.success("Success! Reconstruction is bit-perfect (ignoring floating point epsilon).")

    with tab_details:
        st.subheader("Detail Inspection")
        st.markdown("Zooming into $L_0$ (Fine) vs coarser levels.")
        
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            st.markdown("**Level 0 Detail (Finest)**")
            disp0 = lp[0]
            disp0 = (disp0 - disp0.min()) / (disp0.max() - disp0.min() + 1e-8) * 255
            st.image(disp0.astype(np.uint8), use_container_width=True)
            st.caption("Captures: Point noise, fine textures, sharp edges.")
            
        with d_col2:
            st.markdown(f"**Level {levels-2} Detail (Coarser)**")
            dispN = lp[levels-2]
            dispN = (dispN - dispN.min()) / (dispN.max() - dispN.min() + 1e-8) * 255
            st.image(dispN.astype(np.uint8), use_container_width=True)
            st.caption("Captures: Broad shapes, slow transitions.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Laplacian Pyramid**: Stores detail differences $L_i = G_i - \text{Upsample}(G_{i+1})$.
    - **Visualization**: Detail images look like gray-background edge maps.
    - **Reconstruction**: Add detail back to upsampled coarser levels to recover truth.
    - **Apps**: Compression (most values $\approx 0$), Blending (merge at different scales).
    """)
