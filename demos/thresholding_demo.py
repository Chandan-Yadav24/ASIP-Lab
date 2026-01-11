import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
import io

def otsu_numpy(img):
    """Custom Otsu's thresholding implementation using NumPy."""
    pixel_counts, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    # Probabilities and cumulative sums
    weight1 = np.cumsum(pixel_counts)
    weight2 = np.cumsum(pixel_counts[::-1])[::-1]
    
    # Avoid division by zero
    weight1 = np.maximum(weight1, 1e-10)
    weight2 = np.maximum(weight2, 1e-10)
    
    # Category means
    mean1 = np.cumsum(pixel_counts * bin_centers) / weight1
    mean2 = (np.cumsum((pixel_counts * bin_centers)[::-1]) / weight2[::-1])[::-1]
    
    # Between-class variance
    # We ignore the last element to avoid index out of bounds on mean2[1:]
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2
    
    index_of_max_val = np.argmax(inter_class_variance)
    threshold = bin_centers[:-1][index_of_max_val]
    return threshold

def adaptive_numpy(img, block_size, offset):
    """Custom Adaptive thresholding (local mean) using SciPy's uniform_filter."""
    # Ensure block_size is odd for center alignment
    if block_size % 2 == 0: block_size += 1
    local_mean = uniform_filter(img.astype(float), size=block_size)
    return (img > (local_mean - offset)).astype(np.uint8) * 255

@st.cache_data
def load_img_threshold(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A smooth gradient + shapes + some uneven lighting simulation
        x = np.linspace(0, 1, 512)
        y = np.linspace(0, 1, 512)
        X, Y = np.meshgrid(x, y)
        img_np = (X * 255).astype(np.uint8)
        # Add a "document-like" text block (dark shapes)
        img_np[100:200, 100:400] = 50
        img_np[250:350, 150:350] = 80
        # Add a simulated shadow (multiply by gradient)
        shadow = np.linspace(1.0, 0.4, 512)
        img_np = (img_np * shadow[:, np.newaxis]).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)

def run():
    st.header("⚖️ Thresholding: From Pixels to Decisions")
    st.markdown("""
    Thresholding converts a grayscale image into a **Binary** (B/W) image, 
    separating foreground objects from the background.
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: Segmentation Strategies", expanded=False):
        st.markdown(r"""
        ### 1. The Grayscale-to-Binary Rule
        $B(x, y) = 1$ if $I(x, y) > T$ else $0$
        - $I(x, y)$: Input intensity.
        - $T$: The decision Threshold.

        ### 2. Core Strategies
        - **Global (Simple)**: One fixed $T$ for the whole image. Fast, but fails under uneven lighting.
        - **Otsu's Method**: Automatically finds the **Optimal $T$** by maximizing variance between the two classes (foreground/background).
        - **Adaptive (Local)**: Threshold varies for each pixel based on its local neighborhood. Perfect for removing shadows.

        ### 3. The Stone Sorter Analogy 🪨
        - **Global**: A fixed 1kg limit for every bucket.
        - **Adaptive**: Adjusting the limit based on the size of the local pile.
        - **Otsu's**: Measuring all stones first, then calculating the perfect cut-off point.
        """)

    # --- Global Input ---
    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'bmp', 'webp'], key="thresh_local")
    
    img_gray = load_img_threshold(local_up.read() if local_up else None)
    
    h_orig, w_orig = img_gray.shape
    st.info(f"📁 **Active Image**: `{w_orig}x{h_orig}` | Format: {'Uploaded' if local_up else 'Simulated Document (with shadow)'}")

    tab_global, tab_adaptive, tab_stats = st.tabs(["🧪 Global & Otsu Lab", "🧩 Adaptive (Local) Lab", "📊 Method Comparison"])

    with tab_global:
        st.subheader("Simple vs Automatic Optimal Threshold")
        
        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            method = st.radio("Thresholding Method", ["Manual Slider", "Otsu's Method (Auto)"])
            
            if method == "Manual Slider":
                t_val = st.slider("Global Threshold T", 0, 255, 127)
            else:
                # Calculate Otsu
                try:
                    t_val = otsu_numpy(img_gray)
                    st.success(f"Optimal T found: **{t_val:.0f}**")
                except:
                    t_val = 127
                    st.error("Otsu calculation failed.")
            
            # Binary result
            binary_g = (img_gray > t_val).astype(np.uint8) * 255

        with col_g2:
            st.image(binary_g, caption=f"Result (T={t_val:.0f})", use_container_width=True)
            
        st.divider()
        st.image(img_gray, caption="Original Grayscale", use_container_width=True)
        st.info("💡 **Observation**: Move the slider! Global thresholding often loses detail in shadowed areas (bottom/right).")

    with tab_adaptive:
        st.subheader("Adaptive (Local) Segmentation")
        st.markdown("Thresholding based on the **local neighborhood**. Excellent for uneven lighting.")
        
        col_a1, col_a2 = st.columns([1, 2])
        with col_a1:
            block_size = st.slider("Window Size (Block)", 3, 99, 35, step=2)
            offset_val = st.slider("Offset Constant", -50, 50, 10)
            
            # Adaptive Threshold
            binary_a = adaptive_numpy(img_gray, block_size, offset_val)
            
        with col_a2:
            st.image(binary_a, caption=f"Adaptive (Block: {block_size})", use_container_width=True)
            
        st.warning("🧩 **Local Power**: Notice how Adaptive thresholding can 'see' the text even inside the shadow where Global methods fail!")

    with tab_stats:
        st.subheader("📊 Methodology Comparison")
        
        # Comparison Table
        st.table({
            "Method": ["Global", "Otsu's", "Adaptive"],
            "Threshold Type": ["Fixed (User-set)", "Optimal (Histogram-based)", "Variable (Local Stats)"],
            "Best When": ["Uniform Lighting", "High Contrast Images", "Uneven/Shadowed Media"],
            "Drawback": ["Fails with Shadows", "Fails if not Bimodal", "Slower computation"]
        })
        
        # Small histogram for visual context
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(img_gray.ravel(), bins=64, color='gray', alpha=0.7)
        try:
            o_val = otsu_numpy(img_gray)
            ax.axvline(o_val, color='red', linestyle='--', label=f"Otsu T={o_val:.0f}")
        except: pass
        ax.set_title("Intensity Histogram & Otsu's Pivot")
        ax.set_xlabel("Intensity")
        ax.legend()
        st.pyplot(fig)

    # Final Summary
    st.divider()
    st.markdown("### 📋 Thresholding Profile")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT23-o-yD_073N948M66w7_0_6D_9_X_p_Q&s", width=500) # Placeholder or local if available
    st.caption("Thresholding acts as the final decision layer in many vision systems.")
