# demos/gamma_transform_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

@st.cache_data
def load_img_gamma(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
    else:
        # Default: A smooth linear gradient to show compression/expansion
        x = np.linspace(0, 1, 512)
        y = np.linspace(0, 1, 512)
        X, Y = np.meshgrid(x, y)
        img_np = (X * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
    return np.array(img)

@st.cache_data
def apply_gamma_transform(img, gamma, c):
    # s = c * r^gamma
    r = img.astype(float) / 255.0
    s = c * (r ** gamma)
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)

def run():
    st.header("⚡ Power-law (Gamma) Transformations")
    st.markdown("""
    Gamma transformations are versatile mappings used for image enhancement and device calibration. 
    Unlike Log transforms, they offer a **family of curves** for precise control.
    """)

    # --- Theory Section ---
    with st.expander("📚 Theory: The Power of Exponents", expanded=False):
        st.markdown(r"""
        ### 1. Mathematical Form
        $s = c \cdot r^\gamma$
        - $r$: Input intensity.
        - $s$: Output intensity.
        - $\gamma$: Gamma Exponent (controls the curve).

        ### 2. The Role of $\gamma$
        - **$\gamma < 1$ (Brighten)**: Expands dark values. Excellent for underexposed images.
        - **$\gamma > 1$ (Darken)**: Compresses dark values. Fixes washed-out images with too much glare.
        - **$\gamma = 1$ (Identity)**: Output equals input ($s=r$).

        ### 3. Gamma Correction (Calibration) 🖥️
        Most monitors don't respond linearly to electricity. They naturally have a gamma of $\approx 2.2$. 
        We use an "Inverse Gamma" ($1/2.2$) to pre-correct images so they look correct on your screen.

        ### 4. Mental Model: The Non-linear Dimmer 💡
        Imagine a light switch where turning it halfway only gives 10% brightness. 
        Gamma correction is the mathematical "recalibration" that ensures the physical switch position matches your visual expectation.
        """)

    # --- Global Input Area ---
    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg', 'bmp', 'webp'], key="gamma_local")
    
    img_gray = load_img_gamma(local_up.read() if local_up else None)
    
    h_orig, w_orig = img_gray.shape
    st.info(f"📁 **Active Image**: `{w_orig}x{h_orig}` | Format: {'Uploaded' if local_up else 'Sample Gradient'}")

    tab_lab, tab_curves, tab_correction = st.tabs(["🧪 Interactive Lab", "📉 Family of Curves", "📺 Device Correction"])

    with tab_lab:
        st.subheader("Enhancement Playground")
        col_c1, col_c2 = st.columns([1, 2])
        
        with col_c1:
            st.markdown("**Parameters**")
            gamma_val = st.slider("Gamma Exponent (γ)", 0.05, 5.0, 1.0, step=0.05)
            c_val = st.slider("Constant (c)", 0.1, 2.0, 1.0, step=0.1)
            
            if gamma_val < 1.0:
                st.success("Target: Brightening (Expands Shadows)")
            elif gamma_val > 1.0:
                st.warning("Target: Darkening (Suppresses Glare)")
            else:
                st.info("Target: Identity (No Change)")

        res_gamma = apply_gamma_transform(img_gray, gamma_val, c_val)
        
        with col_c2:
            st.image(res_gamma, caption=f"Result (γ={gamma_val}, c={c_val})", use_container_width=True)

        st.divider()
        c_p1, c_p2 = st.columns(2)
        c_p1.image(img_gray, caption="Original Image", use_container_width=True)
        # Intensity profile (Histogram-like insight)
        fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
        ax_hist.hist(img_gray.ravel(), bins=50, alpha=0.5, label='Original', color='gray')
        ax_hist.hist(res_gamma.ravel(), bins=50, alpha=0.5, label='Transformed', color='tab:blue')
        ax_hist.set_title("Intensity Distribution Shift")
        ax_hist.legend()
        c_p2.pyplot(fig_hist)

    with tab_curves:
        st.subheader("📊 The Mapping Function")
        st.markdown("This curve shows how input intensities (r) map to output intensities (s).")
        
        r_range = np.linspace(0, 1, 256)
        s_current = c_val * (r_range ** gamma_val)
        
        fig_c, ax_c = plt.subplots(figsize=(8, 4))
        # Standard curves for comparison
        ax_c.plot(r_range * 255, (r_range**0.4)*255, 'r--', alpha=0.3, label="γ=0.4 (Bright)")
        ax_c.plot(r_range * 255, (r_range**2.5)*255, 'b--', alpha=0.3, label="γ=2.5 (Dark)")
        ax_c.plot([0, 255], [0, 255], 'k--', alpha=0.2, label="Identity (γ=1.0)")
        
        # Current curve
        ax_c.plot(r_range * 255, np.clip(s_current, 0, 1) * 255, color='tab:blue', lw=4, label=f"Current (γ={gamma_val})")
        
        ax_c.set_xlim(0, 255)
        ax_c.set_ylim(0, 255)
        ax_c.set_xlabel("Input Intensity (r)")
        ax_c.set_ylabel("Output Intensity (s)")
        ax_c.legend()
        ax_c.grid(alpha=0.3)
        st.pyplot(fig_c)

    with tab_correction:
        st.subheader("📺 Display Correction Scenario")
        st.markdown("""
        Most raw digital images assume a linear response, but displays darken them. 
        We 'Gamma Correct' the data by applying **1/γ** before sending it to the screen.
        """)
        
        device_gamma = st.slider("Typical Device Gamma", 1.0, 3.0, 2.2, step=0.1)
        correction = 1.0 / device_gamma
        
        st.latex(rf"s = r^{{1/{device_gamma}}} = r^{{{correction:.2f}}}")
        
        c_corr1, c_corr2 = st.columns(2)
        c_corr1.image(img_gray, caption="Original (Linear Data)", use_container_width=True)
        c_corr2.image(apply_gamma_transform(img_gray, correction, 1.0), 
                    caption=f"Gamma Corrected (for γ={device_gamma})", use_container_width=True)
        
        st.success(f"Applying γ={correction:.2f} compensates for the device's darkening effect!")

    # Summary Table
    st.divider()
    st.markdown("### 📋 Power-law Summary")
    st.table({
        "γ Value": ["γ < 1", "γ = 1", "γ > 1"],
        "Dark Pixels": ["Expanded", "Identity", "Compressed"],
        "Visual Result": ["Brighter Shadows", "True to Source", "Deeper Highlights"],
        "Common Use": ["Underexposed Fix", "Identity Map", "Display Glare Fix"]
    })
