# demos/histogram_processing_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

@st.cache_data
def load_img_hist(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A low-contrast image (mid-range pixels)
        x = np.linspace(0, 5, 512)
        y = np.linspace(0, 5, 512)
        X, Y = np.meshgrid(x, y)
        # Squeezed signal (low contrast)
        img_np = (128 + 30 * (np.sin(X) + np.cos(Y))).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)

def global_equalization(img):
    """Manual Global Histogram Equalization using NumPy."""
    # 1. Calculate histogram
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # 2. Calculate CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max() # for plotting
    
    # 3. Mask zeros to avoid unwanted shifts, then normalize
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 4. Map the pixels
    img_equalized = cdf_final[img]
    
    return img_equalized, hist, cdf_final

def run():
    st.header("📊 Histogram Processing: The statistical Side of Pixels")
    st.markdown("""
    A histogram shows how pixel intensities are distributed. **Histogram Processing** manipulates these distributions to reveal hidden details.
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: Analyzing Distributions", expanded=False):
        st.markdown(r"""
        ### 1. The Basic Concept
        A histogram is a graph showing the count of pixels at each intensity level ($0$ to $255$).
        
        ### 2. Key Techniques
        - **Stretching**: Expands the range to use full spectrum ($0-255$).
        - **Equalization**: Flattens the histogram to maximize contrast.
        - **Matching**: Adjusts one image to look like another's distribution.

        ### 3. The Jar of Coins Analogy 🪙
        Imagine a jar where all coins are jammed in one corner (low contrast). 
        - **Histogram Equalization** is like spreading the coins across the whole floor. 
        - Suddenly, you can see every single coin clearly because they aren't overlapping anymore!

        ### 4. Thresholding Link
        Histograms reveal **Peaks** (Objects) and **Valleys** (Ideal Threshold points). Equalization "pops" these valleys, making it easier to separate foreground from background.
        """)

    # --- Global Input Area ---
    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'bmp', 'webp'], key="hist_local")
    
    img_gray = load_img_hist(local_up.read() if local_up else None)
    
    h_orig, w_orig = img_gray.shape
    st.info(f"📁 **Active Image**: `{w_orig}x{h_orig}` | Format: {'Uploaded' if local_up else 'Sample Mid-tone'}")

    tab_equalize, tab_threshold, tab_analogy = st.tabs(["🧪 Equalization Lab", "⚖️ Thresholding Prep", "🫙 The Jar Analogy"])

    with tab_equalize:
        st.subheader("Global Histogram Equalization (GHE)")
        
        img_eq, hist_orig, cdf_map = global_equalization(img_gray)
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.image(img_gray, caption="Original (Low Contrast)", use_container_width=True)
            # Plot Original Histogram
            fig_h1, ax_h1 = plt.subplots(figsize=(8, 4))
            ax_h1.hist(img_gray.ravel(), bins=256, range=(0, 256), color='gray')
            ax_h1.set_title("Original Histogram: Squeezed")
            st.pyplot(fig_h1)
            
        with col_e2:
            st.image(img_eq, caption="Equalized (Max Contrast)", use_container_width=True)
            # Plot Equalized Histogram
            fig_h2, ax_h2 = plt.subplots(figsize=(8, 4))
            ax_h2.hist(img_eq.ravel(), bins=256, range=(0, 256), color='tab:blue')
            ax_h2.set_title("Equalized Histogram: Flattened")
            st.pyplot(fig_h2)

        st.success("✅ **Insight**: Notice how the histogram spreads across the entire 0-255 range after equalization.")

    with tab_threshold:
        st.subheader("Optimizing for Thresholding")
        st.markdown("""
        Equalization separates overlapping intensities, making it easier for thresholding to 'pop' the text.
        """)
        
        t_val = st.slider("Select Threshold T", 0, 255, 127)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            bin_orig = (img_gray > t_val).astype(np.uint8) * 255
            st.image(bin_orig, caption="Thresholding Original", use_container_width=True)
            st.caption("Fails due to overlap.")
        with col_t2:
            bin_eq = (img_eq > t_val).astype(np.uint8) * 255
            st.image(bin_eq, caption="Thresholding Equalized", use_container_width=True)
            st.caption("Details pop out clearly!")

    with tab_analogy:
        st.subheader("🫙 The Jar of Coins")
        st.markdown("""
        Visualizing the **Spreading Effect**.
        """)
        col_a1, col_a2 = st.columns(2)
        col_a1.metric("Dynamic Range (Orig)", f"{np.min(img_gray)} - {np.max(img_gray)}")
        col_a2.metric("Dynamic Range (Equalized)", "0 - 255")
        
        st.markdown("""
        **How it works**:
        1. We compute the cumulative probability (CDF).
        2. We map each pixel to a new value based on its 'rank' in the distribution.
        3. The resulting image uses all available colors equally.
        """)
        
        # Plot CDF (Mapping Function)
        fig_cdf, ax_cdf = plt.subplots(figsize=(8, 4))
        ax_cdf.plot(cdf_map, color='orange', lw=4)
        ax_cdf.set_title("The Mapping Profile (CDF)")
        ax_cdf.set_xlabel("Input Intensity")
        ax_cdf.set_ylabel("Output Intensity")
        ax_cdf.grid(alpha=0.3)
        st.pyplot(fig_cdf)

    # Summary Table
    st.divider()
    st.markdown("### 📋 Histogram Processing Profile")
    st.table({
        "Technique": ["Equalization", "Stretching", "Matching"],
        "Mechanism": ["CDF mapping", "Min-Max linear expansion", "Mapping to target PDF"],
        "Best When": ["Maximize contrast", "Normalize range", "Lighting consistency"]
    })
