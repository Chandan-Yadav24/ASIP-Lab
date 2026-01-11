# demos/hough_circle_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform, color, data, util, exposure
from PIL import Image
import io

@st.cache_data
def load_img_hough_circle(file_data):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        # Coins are the perfect classic image for circle Hough
        img_np = data.coins()
    return img_np

def run():
    st.header("⭕ 4.3.2 Hough Transform for Circle Detection")
    st.markdown("""
    While the line Hough transform uses $(\\rho, \\theta)$, the Circle Hough transform searches for three parameters: $(a, b, r)$.
    """)

    with st.expander("📚 Theory: The 3D Parameter Space", expanded=False):
        st.markdown(r"""
        ### 1. The Circle Model
        A circle is defined by its center $(a, b)$ and radius $r$:
        $$(x - a)^2 + (y - b)^2 = r^2$$
        Because there are **three** unknowns, the accumulator space is **3D**.
        
        ### 2. The Voting Process
        - **Point $\rightarrow$ Cones**: An edge point $(x, y)$ votes for all possible centers $(a, b)$ at all possible radii $r$. 
        - **Intersection**: If many points lie on a circle, their cones intersect at the 3D point $(a, b, r)$.
        
        ### 3. Efficiency Tricks
        - **Radius Restriction**: Instead of voting for all radii, we limit the search to a range $[R_{min}, R_{max}]$.
        - **Gradient Direction**: Use the edge gradient to vote only for centers along the perpendicular line.
        
        ### 🛶 Ropes around a Hidden Pond Analogy
        Imagine people standing around a circular pond obscured by fog. Each person has a rope (radius). If they all draw circles around themselves with their ropes, the place where most circles overlap is the true center of the pond!
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="circle_local")
    img = load_img_hough_circle(local_up.read() if local_up else None)
    img_gray = util.img_as_float(img)

    st.sidebar.markdown("### 🧰 Parameters")
    min_radius = st.sidebar.slider("Min Radius", 1, 100, 20)
    max_radius = st.sidebar.slider("Max Radius", 10, 150, 50)
    threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.4)

    tab_det, tab_comparison, tab_table = st.tabs(["🎯 Circle Detection Lab", "🆚 Dimensionality Comparison", "📊 Summary Table"])

    # Preprocessing
    edges = feature.canny(img_gray, sigma=3, low_threshold=0.1, high_threshold=0.2)

    with tab_det:
        st.subheader("Detecting Circular Objects")
        st.markdown(f"Searching for radii from **{min_radius}** to **{max_radius}**.")
        
        # Hough Circle Transform
        h_radii = np.arange(min_radius, max_radius + 1, 2)
        hough_res = transform.hough_circle(edges, h_radii)
        
        # Select the best circles
        accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, h_radii, total_num_peaks=10, threshold=threshold)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        for center_y, center_x, radius in zip(cy, cx, radii):
            circ = plt.Circle((center_x, center_y), radius, color='yellow', fill=False, linewidth=2)
            ax.add_patch(circ)
            ax.plot(center_x, center_y, 'r+', markersize=10)
            
        ax.set_title(f"Detected {len(radii)} Circles")
        ax.axis('off')
        st.pyplot(fig)
        
        st.info("Yellow circles indicate the detected boundaries, and red crosses mark the centers $(a, b)$.")

    with tab_comparison:
        st.subheader("Why Circle Hough is 'Heavier'")
        
        st.table({
            "Feature": ["Line Hough", "Circle Hough"],
            "Parameters": ["2 (ρ, θ)", "3 (a, b, r)"],
            "Accumulator": ["2D Grid", "3D Volume"],
            "Memory Complexity": ["Low", "High (exponentially higher)"],
            "Constraint Required": ["Max Distance", "Radius Range [min, max]"]
        })
        
        st.warning("Because of the 3D accumulator, detecting circles requires much more memory and processing power than lines.")

    with tab_table:
        st.subheader("Circle Hough Framework")
        st.table({
            "Stage": ["Preprocessing", "Radius Range", "3D Voting", "Peak Finding", "Extraction"],
            "Goal": [
                "Extract edge map (Canny)",
                "Limit the search space",
                "Increment cells in A(a,b,r)",
                "Identify local maxima",
                "Map (a,b,r) back to image"
            ]
        })

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **Model**: $(x-a)^2 + (y-b)^2 = r^2$.
    - **Complexity**: 3D Accumulator makes it computationally expensive.
    - **Optimization**: Always restrict the radius range to save memory.
    - **Advantage**: Detects circles even with gaps or overlapping objects (like coins).
    """)
