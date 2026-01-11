# demos/histogram_equalization_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

@st.cache_data
def load_img_he(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A very low contrast sample
        x = np.linspace(0, 10, 512)
        y = np.linspace(0, 10, 512)
        X, Y = np.meshgrid(x, y)
        # Squeezed signal 
        img_np = (100 + 20 * (np.sin(X) + np.cos(Y))).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)

def global_he_numpy(img):
    """CDF-based Global Histogram Equalization."""
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    # Mask zeros and normalize
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    return cdf_final[img], cdf_final

def adaptive_he_tiled(img, grid_size=(8, 8)):
    """Simple tiled Adaptive HE (no interpolation to keep it fast/simple for demo)."""
    h, w = img.shape
    gh, gw = grid_size
    th, tw = h // gh, w // gw
    
    res = np.zeros_like(img)
    for i in range(gh):
        for j in range(gw):
            tile = img[i*th:(i+1)*th, j*tw:(j+1)*tw]
            # Local equalization on tile
            hist, _ = np.histogram(tile.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_m = np.ma.masked_equal(cdf, 0)
            if cdf_m.max() > cdf_m.min():
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
            res[i*th:(i+1)*th, j*tw:(j+1)*tw] = cdf_final[tile]
            
    return res

def run():
    st.header("⚖️ Histogram Equalization (HE): Automatic Contrast")
    st.markdown("""
    HE is a data-driven transformation thatорганизует pixel intensities to use the full range [0, 255] as evenly as possible.
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: The Math of Spreading", expanded=False):
        st.markdown(r"""
        ### 1. Probability Density Function (PDF)
        For an image of size $M \times N$, the probability of a gray level $r_k$ is:
        $p(r_k) = \frac{n_k}{MN}$
        
        ### 2. The CDF Transform
        The new intensity $s_k$ is calculated using the Cumulative Distribution Function (CDF):
        $s_k = (L-1) \sum_{j=0}^{k} p(r_j)$
        where $L$ is the number of gray levels (usually 256).

        ### 3. Key Characteristics
        - **Automatic**: No manual input needed.
        - **Monotonic**: Preserves relative brightness (blacks stay darker than whites).
        - **Side Effects**: Can amplify noise or create intensity gaps.

        ### 4. The Crowded Room Analogy 👥
        Imagine a party where everyone is huddled in a single dark corner. Identifying individuals is hard. 
        **Equalization** organizes everyone to spread out across the entire room. Now, people are much easier to see because the "space" between them is maximized!
        """)

    # --- Global Input Area ---
    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="he_local")
    
    img_gray = load_img_he(local_up.read() if local_up else None)
    
    h_orig, w_orig = img_gray.shape
    st.info(f"📁 **Active Image**: `{w_orig}x{h_orig}` | Format: {'Uploaded' if local_up else 'Sample Mid-tone'}")

    tab_compare, tab_math, tab_metrics = st.tabs(["🧪 Global vs Adaptive Lab", "📐 Mathematical Profiling", "📋 HE Characteristics"])

    with tab_compare:
        st.subheader("GHE vs. Tiled Adaptive HE")
        
        col_c1, col_c2 = st.columns([1, 2])
        with col_c1:
            method = st.radio("Select Strategy", ["Global HE", "Tiled Adaptive HE"])
            if method == "Global HE":
                res_img, _ = global_he_numpy(img_gray)
                st.caption("Strategy: Compute one mapping for the whole image.")
            else:
                grid = st.slider("Grid Size (Tiles)", 2, 24, 8)
                res_img = adaptive_he_tiled(img_gray, (grid, grid))
                st.caption(f"Strategy: Compute independent mappings for {grid}x{grid} local blocks.")
        
        with col_c2:
            st.image(res_img, caption=f"Result: {method}", use_container_width=True)
            
        st.divider()
        c_p1, c_p2 = st.columns(2)
        c_p1.image(img_gray, caption="Original (Unprocessed)", use_container_width=True)
        # Histogram comparison
        fig_h, ax_h = plt.subplots(figsize=(8, 3))
        ax_h.hist(img_gray.ravel(), bins=64, alpha=0.4, label='Original', color='gray')
        ax_h.hist(res_img.ravel(), bins=64, alpha=0.4, label='Processed', color='tab:blue')
        ax_h.set_title("Histogram Transformation")
        ax_h.legend()
        c_p2.pyplot(fig_h)

    with tab_math:
        st.subheader("The CDF Mapping Logic")
        st.markdown("This curve is derived from your image's statistics ($s_k = T(r_k)$).")
        
        _, cdf_curve = global_he_numpy(img_gray)
        
        fig_m, ax_m = plt.subplots(figsize=(8, 4))
        ax_m.plot(cdf_curve, color='tab:green', lw=4, label='Calculated CDF Mapping')
        ax_m.plot([0, 255], [0, 255], 'k--', alpha=0.3, label='Linear (No change)')
        ax_m.set_xlabel("Input Intensity (r)")
        ax_m.set_ylabel("Output Intensity (s)")
        ax_m.set_xlim(0, 255)
        ax_m.set_ylim(0, 255)
        ax_m.grid(alpha=0.3)
        ax_m.legend()
        st.pyplot(fig_m)
        
        st.info("💡 **Math Tip**: The steeper the CDF curve at a specific point, the more the intensities in that range are being 'pulled apart' (expanded).")

    with tab_metrics:
        st.subheader("Comparison Summary")
        st.table({
            "Item": ["Global HE", "Adaptive HE (CLAHE)"],
            "Scope": ["Entire Image Histogram", "Local Neighborhoods/Blocks"],
            "Strength": ["Automated overall contrast", "Reveals hidden local details"],
            "Risk": ["Can wash out small details", "Noise amplification (over-sharpening)"],
            "Analogy": ["Everyone spreads across one room", "Each group spreads in their own corner"]
        })

    # Summary Table
    st.divider()
    st.markdown("### 📋 Histogram Equalization Profile")
    st.markdown("""
    | Property | Description |
    | :--- | :--- |
    | **Point-wise** | Each pixel is processed based on its value and image statistics. |
    | **Automatic** | Purely data-driven; no threshold needed. |
    | **Monotonic** | Order of brightness is preserved. |
    | **Artifacts** | Can cause 'false contouring' if gaps are too large. |
    """)
