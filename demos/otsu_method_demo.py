# demos/otsu_method_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, data, util, exposure, color
from PIL import Image
import io

@st.cache_data
def load_img_otsu(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Coins/Camera are classic for bimodal Otsu testing
        img_np = data.coins()
    return img_np

def compute_otsu_variance(img):
    """Manual computation of between-class variance for every possible threshold."""
    hist, bin_edges = np.histogram(img.ravel(), bins=256, range=(0, 255))
    hist = hist.astype(float) / hist.sum() # Probability
    
    thresholds = np.arange(256)
    variances = np.zeros(256)
    
    # Global mean
    mu_total = np.sum(thresholds * hist)
    
    for t in thresholds:
        # Background class (0 to t)
        w1 = np.sum(hist[:t+1])
        if w1 == 0: continue
        mu1 = np.sum(thresholds[:t+1] * hist[:t+1]) / w1
        
        # Foreground class (t+1 to 255)
        w2 = 1.0 - w1
        if w2 == 0: continue
        mu2 = np.sum(thresholds[t+1:] * hist[t+1:]) / w2
        
        # Between-class variance
        variances[t] = w1 * w2 * (mu1 - mu2)**2
        
    return thresholds, variances, mu_total

def run():
    st.header("📊 4.11 Otsu’s Method (Automatic Thresholding)")
    st.markdown("""
    Otsu's method is a classic algorithm that automatically finds the best threshold to separate an image into background and foreground based on variance optimization.
    """)

    with st.expander("📚 Theory: The Principle of Optimality", expanded=False):
        st.markdown(r"""
        ### 1. The Goal
        Maximize the **between-class variance** ($\sigma_B^2$). This is equivalent to finding the threshold that makes the two resulting groups (classes) as distinct as possible.
        
        ### 2. The Logic
        - **Between-class variance** measures how spread apart the means of the two groups are.
        - **Algorithm**:
          1. Compute image histogram.
          2. Calculate cumulative sums and means for every possible threshold $k$.
          3. Compute $\sigma_B^2(k) = w_1(k) w_2(k) [\mu_1(k) - \mu_2(k)]^2$.
          4. Pick $k^*$ where $\sigma_B^2$ is maximum.
          
        ### 📏 Separability Metric ($\eta$)
        Otsu can also tell you how "separable" your image is:
        $$\eta = \frac{\sigma_B^2(k^*)}{\sigma_{Global}^2}$$
        A high $\eta$ means a clear cut (e.g., black text on white paper).
        
        ### 👯 Analogy: Heights in a Room
        Imagine you want to split a group of people into "Short" and "Tall" using one height cutoff. Otsu's method tests every possible height and picks the one that creates the most consistent and separate groups.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="otsu_local")
    img = load_img_otsu(local_up.read() if local_up else None)
    img_float = util.img_as_float(img)

    tab_math, tab_visual, tab_multi, tab_table = st.tabs([
        "🔬 Optimization Math", "🖼️ Threshold Result", "🪜 Multi-level Otsu", "📊 Quick Summary"
    ])

    # Compute manual Otsu stats
    thresholds, variances, mu_total = compute_otsu_variance(img)
    k_star = np.argmax(variances)
    eta = variances[k_star] / np.var(img)

    with tab_math:
        st.subheader("Visualizing Between-Class Variance")
        st.markdown("We search for the peak of the variance curve to find the optimal threshold.")
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Histogram
        ax2 = ax1.twinx()
        ax2.hist(img.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.3, label='Histogram')
        ax2.set_ylabel("Pixel Count")
        
        # Variance Curve
        ax1.plot(thresholds, variances, 'r-', linewidth=2, label='Between-Class Variance')
        ax1.axvline(k_star, color='blue', linestyle='--', label=f'Peak at {k_star}')
        ax1.set_xlabel("Potential Threshold (k)")
        ax1.set_ylabel("Variance Value")
        ax1.set_title("Variance Maximization vs Histogram")
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        st.pyplot(fig)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Threshold", k_star)
        col2.metric("Separability (η)", f"{eta:.3f}")
        col3.metric("Global Mean", f"{mu_total:.1f}")

    with tab_visual:
        st.subheader("Automatic Binary Result")
        binary = img > k_star
        
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original Image", use_container_width=True)
        col2.image(util.img_as_ubyte(binary), caption=f"Otsu Binary (T={k_star})", use_container_width=True)
        st.success(f"Otsu automatically picked {k_star} based on the variance peak seen in the previous tab.")

    with tab_multi:
        st.subheader("Multi-threshold Otsu")
        st.markdown("Extending the method to find more than two classes (e.g., Water, Ice, Land).")
        
        n_levels = st.select_slider("Select Number of Classes", options=[2, 3, 4], value=3)
        
        if n_levels == 2:
            multi_thresh = [filters.threshold_otsu(img)]
        else:
            multi_thresh = filters.threshold_multiotsu(img, classes=n_levels)
            
        regions = np.digitize(img, bins=multi_thresh)
        
        fig_m, ax_m = plt.subplots()
        ax_m.imshow(regions, cmap='viridis')
        ax_m.set_title(f"{n_levels}-Class Segmentation")
        ax_m.axis('off')
        st.pyplot(fig_m)
        st.info(f"Thresholds found at: {', '.join([str(int(t)) for t in multi_thresh])}")

    with tab_table:
        st.subheader("Otsu's Method Framework")
        st.table({
            "Stage": ["Probability", "Class Weight", "Class Mean", "Variance", "Selection"],
            "Definition": [
                "Normalized histogram P(i)",
                "Sum of P(i) for each group",
                "Average intensity of each group",
                "Distance between group means",
                "Maximize between-class variance"
            ],
            "Goal": [
                "Map distributions",
                "Count members in G1 vs G2",
                "Find 'center' of each class",
                "Measure group separation",
                "The 'Principle of Optimality'"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Optimality**: Choose threshold that maximizes between-class variance.
    - **Equivalent**: This also minimizes within-class variance.
    - **Condition**: Works best for **bimodal** histograms (two clear peaks).
    - **Parameter-free**: Fully automatic; no user input needed for the cutoff.
    """)
