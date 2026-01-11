# demos/advanced_thresholding_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, data, util, exposure, color
from skimage.morphology import disk
from PIL import Image
import io

@st.cache_data
def load_img_threshold(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Text/Page image is perfect for demonstrating thresholding challenges
        img_np = data.page()
    return img_np

def iterative_threshold(img, max_iter=10):
    t = np.mean(img)
    history = [t]
    for _ in range(max_iter):
        g1 = img[img > t]
        g2 = img[img <= t]
        if len(g1) == 0 or len(g2) == 0:
            break
        m1 = np.mean(g1)
        m2 = np.mean(g2)
        new_t = 0.5 * (m1 + m2)
        if abs(new_t - t) < 0.5:
            break
        t = new_t
        history.append(t)
    return t, history

def run():
    st.header("🌡️ 4.10 Advanced Thresholding")
    st.markdown("""
    Thresholding is the process of converting a grayscale image into a binary one to isolate objects from the background.
    """)

    with st.expander("📚 Theory: From Fixed Cutoffs to Adaptive Intelligence", expanded=False):
        st.markdown(r"""
        ### 1. Global Thresholding
        - **Fixed**: A single intensity $T$ for the whole image.
        - **Iterative Selection**: Automatically finds $T$ by splitting pixels into two groups and updating $T$ based on their mean values until convergence.
        
        ### 2. Otsu’s Method (The Optimal Search)
        Instead of guessing, Otsu's method exhaustively searches for the threshold that **maximizes between-class variance**. It effectively finds the "deepest valley" in a bimodal histogram.
        
        ### 3. Adaptive (Variable) Thresholding
        Global methods fail when lighting is uneven. Adaptive thresholding calculates a unique threshold $T(x, y)$ for every pixel based on its local neighborhood.
        
        ### 4. Hysteresis (Canny Logic)
        Uses two thresholds:
        - **High ($T_H$)**: Reliable strong edges.
        - **Low ($T_L$)**: Potential weak edges.
        - **Logic**: Weak edges are kept only if they "touch" a strong edge.
        
        ### 🎭 The Party Judge Analogy
        - **Global**: A judge at the door with one fixed rule for the whole party.
        - **Otsu**: A judge who studies everyone's outfits first and finds the best split.
        - **Adaptive**: A judge who walks around with a torch, using different rules in dark corners vs. bright spots.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="thresh_local")
    img = load_img_threshold(local_up.read() if local_up else None)
    img_float = util.img_as_float(img)

    tab_iter, tab_otsu, tab_adaptive, tab_hysteresis = st.tabs([
        "🔄 Iterative Global", "📊 Otsu's Method", "🌓 Adaptive (Local)", "🧠 Hysteresis"
    ])

    with tab_iter:
        st.subheader("Iterative Threshold Selection")
        st.markdown("Watch the threshold converge automatically based on group means.")
        
        t_final, history = iterative_threshold(img)
        
        col1, col2 = st.columns(2)
        col1.image(util.img_as_ubyte(img > t_final), caption=f"Final Binary (T={t_final:.1f})", use_container_width=True)
        
        fig_conv, ax_conv = plt.subplots()
        ax_conv.plot(history, 'bo-')
        ax_conv.set_title("Threshold Convergence")
        ax_conv.set_xlabel("Iteration")
        ax_conv.set_ylabel("Threshold Value")
        col2.pyplot(fig_conv)
        st.info(f"Threshold started at {history[0]:.1f} and converged to {t_final:.1f} in {len(history)-1} steps.")

    with tab_otsu:
        st.subheader("Otsu's Optimal Variance")
        st.markdown("Finding the best split by analyzing the histogram.")
        
        t_otsu = filters.threshold_otsu(img)
        binary_otsu = util.img_as_ubyte(img > t_otsu)
        
        col1, col2 = st.columns(2)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(img.ravel(), bins=256, color='gray', alpha=0.7)
        ax_hist.axvline(t_otsu, color='red', linestyle='--', label=f'Otsu T={int(t_otsu)}')
        ax_hist.set_title("Image Histogram")
        ax_hist.legend()
        col1.pyplot(fig_hist)
        
        col2.image(binary_otsu, caption="Otsu Output", use_container_width=True)

    with tab_adaptive:
        st.subheader("Adaptive vs Global")
        st.markdown("Useful for unevenly lit documents where one threshold doesn't fit all.")
        
        block_size = st.slider("Local Neighborhood Size", 3, 99, 35, step=2)
        offset = st.slider("Threshold Offset", 0.0, 0.2, 0.03)
        
        t_global = filters.threshold_otsu(img)
        binary_global = util.img_as_ubyte(img > t_global)
        
        binary_adaptive = filters.threshold_local(img_float, block_size, offset=offset)
        binary_adaptive = util.img_as_ubyte(img_float > binary_adaptive)
        
        c1, c2 = st.columns(2)
        c1.image(binary_global, caption="Global (Otsu) - Fails in dark areas", use_container_width=True)
        c2.image(binary_adaptive, caption="Adaptive (Local) - Handles uneven light", use_container_width=True)

    with tab_hysteresis:
        st.subheader("Hysteresis Thresholding")
        st.markdown("Linking weak edges to strong ones to preserve continuity.")
        
        low = st.slider("Low Threshold", 0.0, 0.5, 0.1)
        high = st.slider("High Threshold", 0.1, 0.9, 0.3)
        
        hyst = filters.apply_hysteresis_threshold(img_float, low, high)
        
        c1, c2 = st.columns(2)
        c1.image(util.img_as_ubyte(img_float > high), caption="Strong Edges Only", use_container_width=True)
        c2.image(util.img_as_ubyte(hyst), caption="Hysteresis (Linked Edges)", use_container_width=True)
        st.caption("Notice how Hysteresis fills in gaps by following paths from strong markers.")

    st.divider()
    st.markdown("### 📋 Summary Table")
    st.table({
        "Method": ["Fixed Global", "Iterative", "Otsu", "Adaptive", "Hysteresis"],
        "Best For": ["Ideal conditions", "General auto-select", "Bimodal histograms", "Uneven lighting", "Edge linking"],
        "Limitation": ["No flexibility", "Lighting issues", "Fails on noisy valleys", "Computationally heavy", "Specific to boundaries"]
    })
