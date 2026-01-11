# demos/intensity_transform_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve
import io

@st.cache_data
def load_img_it(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A smooth gradient + some shapes
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        X, Y = np.meshgrid(x, y)
        img_np = (X * 255).astype(np.uint8)
        # Add a white square
        img_np[100:150, 100:150] = 255
        img = Image.fromarray(img_np)
    return np.array(img)

@st.cache_data
def apply_point_op(img, op_type, params):
    r = img.astype(float) / 255.0
    
    if op_type == "Negative":
        s = 1.0 - r
    elif op_type == "Logarithmic":
        c = params.get("c", 1.0)
        s = c * np.log(1 + r)
    elif op_type == "Gamma (Power-law)":
        gamma = params.get("gamma", 1.0)
        c = params.get("c", 1.0)
        s = c * (r ** gamma)
    elif op_type == "Thresholding":
        t = params.get("threshold", 0.5)
        s = (r >= t).astype(float)
    else:
        s = r
        
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)

def run():
    st.header("✨ Intensity Transformations")
    st.markdown("""
    Explore how manipulating individual pixels (**Point Processing**) or small groups of pixels (**Neighborhood Processing**) can transform an image.
    """)

    # Theory Section
    with st.expander("📚 Theory: Spatial Domain Basics", expanded=False):
        st.markdown(r"""
        ### 1. Spatial Domain vs Frequency Domain
        - **Spatial Domain**: Operations done directly on the pixels $f(x,y)$. Fast, simple, pixel-by-pixel.
        - **Frequency Domain**: Operates on the Fourier Transform. Best for pattern removal and compression.

        ### 2. Transformation Model
        $g(x, y) = T[f(x, y)]$
        - $f(x, y)$: Input intensity.
        - $g(x, y)$: Output intensity.
        - $T$: The transformation rule.

        ### 3. Point Processing
        When $T$ only looks at one pixel at a time, we use the simplified form:
        $s = T(r)$
        Where $r$ is the input intensity and $s$ is the output.
        """)

    # --- Global Input Area (Local + Sidebar) ---
    st.sidebar.markdown("---") # Visual separator in sidebar
    
    # Local Uploader for this practical
    local_file = st.file_uploader("📤 Upload Image for Transformations (Overrides Sidebar)", 
                                 type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'], 
                                 key="it_local_upload")
    
    # Global sidebar file from main.py if available (handled via streamlit state or re-upload)
    # Since we are in a sub-module, we look at the specific key if it exists in session state
    # But standard way is to just use the one passed or local.
    
    # Determine the Source Image
    if local_file:
        pil_source = Image.open(local_file)
        source_name = local_file.name
        source_size = local_file.size / 1024
        source_fmt = pil_source.format
        img_gray = np.array(pil_source.convert('L'))
    else:
        # Fallback to sample if no upload
        # (Alternatively we could check if user uploaded in sidebar in another lab, 
        # but standardizing on local lab uploads or sample is cleaner for this specific lab)
        img_gray = load_img_it(None) 
        source_name = "Sample Gradient"
        source_size = 256.0
        source_fmt = "Generated"

    # Display Metadata
    h, w = img_gray.shape
    st.info(f"📁 **Active Image**: `{source_name}` | **Format**: {source_fmt} | **Res**: {w}x{h} | **Size**: {source_size:.1f} KB")

    tab_point, tab_neighborhood, tab_lut = st.tabs(["🧪 Point Processing Lab", "🧩 Neighborhood Lab", "📊 LUT Visualizer"])

    with tab_point:
        st.subheader("Point Processing: Pixel-by-Pixel")
        col_p1, col_p2 = st.columns([1, 2])
        
        with col_p1:
            op_type = st.radio("Select Transformation", 
                              ["Negative", "Logarithmic", "Gamma (Power-law)", "Thresholding"])
            
            p = {}
            if op_type == "Logarithmic":
                p["c"] = st.slider("Constant c", 0.1, 2.0, 1.0)
            elif op_type == "Gamma (Power-law)":
                p["gamma"] = st.slider("Gamma (γ)", 0.05, 5.0, 1.0)
                p["c"] = st.slider("Constant c", 0.1, 2.0, 1.0)
            elif op_type == "Thresholding":
                p["threshold"] = st.slider("Threshold Level", 0.0, 1.0, 0.5)
            
            res_img = apply_point_op(img_gray, op_type, p)
            
        with col_p2:
            st.image(res_img, caption=f"Processed Image ({op_type})", use_container_width=True)
            
        st.divider()
        c_p1, c_p2 = st.columns(2)
        c_p1.image(img_gray, caption="Original Image", use_container_width=True)
        c_p2.markdown(f"""
        **Insight**:  
        Current Function: `{op_type}`  
        The transformation is applied independently to every pixel. No neighborhood context is used here.
        """)

    with tab_neighborhood:
        st.subheader("Neighborhood Concept (3x3 Filtering)")
        st.markdown("""
        Unlike point processing, the new value of a pixel depends on its **Neighbors**.
        Let's apply a 3x3 **Averaging** filter (Smoothing).
        """)
        
        k_size = st.select_slider("Select Neighborhood Size", options=[3, 5, 7, 9], value=3)
        kernel = np.ones((k_size, k_size)) / (k_size**2)
        
        smooth_img = convolve(img_gray.astype(float), kernel).astype(np.uint8)
        
        col_n1, col_n2 = st.columns(2)
        col_n1.image(img_gray, caption="Original (Grainy/Sharp)", use_container_width=True)
        col_n2.image(smooth_img, caption=f"Smoothed ({k_size}x{k_size} Avg)", use_container_width=True)
        
        st.info("💡 **Neighborhood Rule:** We take a window, compute the average of all pixels inside, and assign that to the center. This blurs sharp edges and reduces noise.")

    with tab_lut:
        st.subheader("📊 Look-Up Table (LUT) Mapping")
        st.markdown("""
        For performance ($O(1)$), computers precompute the transformation once for all 256 possible colors and store them in a table.
        """)
        
        # Calculate LUT for current settings
        r_range = np.linspace(0, 1, 256)
        # Mocking apply_point_op logic for the curve
        if op_type == "Negative": s_curve = 1.0 - r_range
        elif op_type == "Logarithmic": s_curve = p["c"] * np.log(1 + r_range)
        elif op_type == "Gamma (Power-law)": s_curve = p["c"] * (r_range ** p["gamma"])
        elif op_type == "Thresholding": s_curve = (r_range >= p["threshold"]).astype(float)
        else: s_curve = r_range
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(r_range * 255, np.clip(s_curve, 0, 1) * 255, color='cyan', lw=3)
        ax.set_title(f"Intensity Mapping: {op_type}")
        ax.set_xlabel("Input Intensity (r)")
        ax.set_ylabel("Output Intensity (s)")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 255)
        ax.set_ylim(-5, 260)
        st.pyplot(fig)
        
        st.caption("This curve shows exactly how each input brightness is remapped to a new output brightness.")

    # Summary Table
    st.divider()
    st.markdown("### 📋 Quick Reference: Point Mapping Functions")
    st.table({
        "Function": ["Negative", "Log", "Gamma", "Threshold"],
        "Math Model": ["s = L - 1 - r", "s = c * log(1+r)", "s = c * r^γ", "s = 0 if r < T else 1"],
        "Visual Use": ["Inverts", "Expands darks", "Contrast fix", "Binary B/W"]
    })
