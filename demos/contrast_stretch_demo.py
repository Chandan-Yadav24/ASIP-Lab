# demos/contrast_stretch_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

@st.cache_data
def load_img_contrast(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A low-contrast image (mid-range pixels only)
        x = np.linspace(0, 1, 512)
        y = np.linspace(0, 1, 512)
        X, Y = np.meshgrid(x, y)
        img_np = (100 + 50 * (np.sin(X*10) + np.cos(Y*10))).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)

@st.cache_data
def piecewise_linear(img, r1, s1, r2, s2):
    # img is 8-bit [0, 255]
    res = np.zeros_like(img, dtype=float)
    r = img.astype(float)
    
    # Range 1: 0 to r1
    mask1 = r <= r1
    if r1 > 0:
        res[mask1] = (s1 / r1) * r[mask1]
    else:
        res[mask1] = s1
        
    # Range 2: r1 to r2
    mask2 = (r > r1) & (r <= r2)
    if r2 > r1:
        res[mask2] = ((s2 - s1) / (r2 - r1)) * (r[mask2] - r1) + s1
    else:
        res[mask2] = s2
        
    # Range 3: r2 to 255
    mask3 = r > r2
    if 255 > r2:
        res[mask3] = ((255 - s2) / (255 - r2)) * (r[mask3] - r2) + s2
    else:
        res[mask3] = s2
        
    return np.clip(res, 0, 255).astype(np.uint8)

@st.cache_data
def auto_min_max(img, clip_percent=0):
    flat = img.ravel()
    if clip_percent > 0:
        low = np.percentile(flat, clip_percent)
        high = np.percentile(flat, 100 - clip_percent)
    else:
        low, high = np.min(flat), np.max(flat)
    
    if high > low:
        res = (img.astype(float) - low) / (high - low) * 255
    else:
        res = img.astype(float)
        
    return np.clip(res, 0, 255).astype(np.uint8)

def run():
    st.header("🎻 Contrast Stretching: The Folded Accordion")
    st.markdown("""
    Contrast stretching is a point-wise transformation that 'pulls open' the intensity range of a squeezed (low-contrast) image.
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: Stretching the Dynamic Range", expanded=False):
        st.markdown(r"""
        ### 1. Why is Contrast Low?
        - **Poor Illumination**: Dim light creates compressed dark tones.
        - **Sensor Limits**: Hardware cannot capture full depth.
        - **Settings**: Incorrect aperture/exposure.

        ### 2. Piecewise Linear Mapping
        We define control points $(r_1, s_1)$ and $(r_2, s_2)$ to reshape the mapping:
        - **$r < r_1$**: Compressed (pushed darker).
        - **$r_1 < r < r_2$**: Stretched (increased contrast).
        - **$r > r_2$**: Compressed (pushed brighter).

        ### 3. The Accordion Analogy 🪗
        A low-contrast image is like a tightly squeezed accordion. Stretching it is like pulling it to its full length, making every 'pleat' (intensity level) visible and distinct.

        ### 4. Rule for Realism
        To avoid "flipping" tones, we maintain:
        $r_1 \leq r_2$ and $s_1 \leq s_2$.
        """)

    # --- Global Input Area ---
    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'bmp', 'webp'], key="contrast_local")
    
    img_gray = load_img_contrast(local_up.read() if local_up else None)
    
    h_orig, w_orig = img_gray.shape
    st.info(f"📁 **Active Image**: `{w_orig}x{h_orig}` | Format: {'Uploaded' if local_up else 'Sample Mid-tone'}")

    tab_manual, tab_auto, tab_histogram = st.tabs(["🧪 Manual Control Lab", "🤖 Auto Min-Max Lab", "📊 Histogram Insight"])

    with tab_manual:
        st.subheader("Piecewise Linear Controls")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Control Points**")
            r1 = st.slider("r1 (Input Low)", 0, 255, 60)
            s1 = st.slider("s1 (Output Low)", 0, 255, 20)
            
            st.divider()
            
            r2 = st.slider("r2 (Input High)", r1, 255, 190)
            s2 = st.slider("s2 (Output High)", s1, 255, 235)
            
            if r1 == r2:
                st.info("💡 Special Case: **Thresholding** profile selected.")
            elif r1 == s1 and r2 == s2:
                st.info("💡 **Identity Map** (No Change).")
            else:
                st.success("Target: Contrast Stretching")

        res_manual = piecewise_linear(img_gray, r1, s1, r2, s2)
        
        with c2:
            st.image(res_manual, caption="Stretched Result", use_container_width=True)

    with tab_auto:
        st.subheader("Automatic Min-Max Scaling")
        st.markdown("Instantly maps the existing `min` to `0` and `max` to `255`.")
        
        clip_p = st.slider("Clip Outliers (%)", 0.0, 5.0, 0.0, step=0.1)
        res_auto = auto_min_max(img_gray, clip_p)
        
        col_a1, col_a2 = st.columns(2)
        col_a1.image(img_gray, caption="Original View", use_container_width=True)
        col_a2.image(res_auto, caption=f"Auto-Stretched (Clip: {clip_p}%)", use_container_width=True)
        
        st.success(f"Original range: [{np.min(img_gray)}, {np.max(img_gray)}] → New range: [0, 255]")

    with tab_histogram:
        st.subheader("📊 Viewing the Spread")
        
        # Mapping curve plot
        r_line = np.arange(256)
        s_line = piecewise_linear(r_line, r1, s1, r2, s2)
        
        fig, (ax_curve, ax_hist) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Curve
        ax_curve.plot(r_line, s_line, color='tab:red', lw=3, label='Mapping Function')
        ax_curve.plot([0, 255], [0, 255], 'k--', alpha=0.2, label='Identity')
        ax_curve.set_title("Piecewise Linear Mapping Profile")
        ax_curve.set_xlim(0, 255)
        ax_curve.set_ylim(0, 255)
        ax_curve.grid(alpha=0.3)
        ax_curve.legend()
        
        # Histogram
        ax_hist.hist(img_gray.ravel(), bins=64, range=(0, 256), alpha=0.5, label='Original', color='gray')
        ax_hist.hist(res_manual.ravel(), bins=64, range=(0, 256), alpha=0.5, label='Manual Stretched', color='tab:red')
        ax_hist.set_title("Histogram Comparison (Dynamic Range Spread)")
        ax_hist.set_xlabel("Intensity")
        ax_hist.legend()
        
        plt.tight_layout()
        st.pyplot(fig)

    # Summary Table
    st.divider()
    st.markdown("### 📋 Contrast Transformation Profile")
    st.table({
        "Feature": ["Goal", "Method", "Special Case", "Auto Variant", "Logic"],
        "Details": ["Expand intensity range", "Piecewise linear mapping", "Thresholding (r1=r2)", "Min-max scaling", "Memory-less O(1)"]
    })
