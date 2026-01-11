# demos/convex_hull_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull as scipy_hull
from skimage import morphology, measure, color
from PIL import Image
import io

@st.cache_data
def load_img_hull(file_data, threshold=128):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
        binary = img_np > threshold
    else:
        # Synthetic object with "dents" (concavities)
        # Create a star or U-shape
        img_np = np.zeros((400, 400), dtype=bool)
        # U-shape
        img_np[100:300, 100:120] = True
        img_np[100:300, 280:300] = True
        img_np[280:300, 100:300] = True
        # Small isolated dots inside out
        img_np[150, 150] = True
        binary = img_np
    return binary

def run():
    st.header("⚪ Convex Hull: The Rubber Band Shape")
    st.markdown("""
    The **Convex Hull** is the smallest convex shape that completely encloses an object. 
    It "fills in" all the dents and concavities.
    """)

    with st.expander("📚 Theory: Nails & Rubber Bands", expanded=False):
        st.markdown(r"""
        ### 1. Intuition
        Imagine your object's pixels are nails in a board. If you stretch a rubber band around all the nails, the shape it takes when tightening is the **Convex Hull**.
        
        ### 2. Algorithms
        - **Graham Scan**: Sorts points by polar angle; $O(N \log N)$.
        - **Jarvis's March**: "Gift Wrapping" algorithm; $O(Nh)$.
        - **QuickHull**: Divide-and-conquer approach; usually $O(N \log N)$.
        
        ### 3. Key Metric: Solidity
        $$Solidity = \frac{\text{Area of Object}}{\text{Area of Convex Hull}}$$
        - **Circle/Square**: Solidity close to 1.0 (very convex).
        - **Star/U-Shape**: Lower solidity (has many "dents").
        """)

    tab_points, tab_image, tab_metric = st.tabs(["🔢 Numerical Set", "🖼️ Object Hull", "📊 Shape Analysis"])

    with tab_points:
        st.subheader("6-Point Challenge")
        st.markdown("Points: (1, 2), (3, 5), (6, 4), (7, 2), (4, 1), (2, 1)")
        
        points = np.array([
            [1, 2], [3, 5], [6, 4], [7, 2], [4, 1], [2, 1]
        ])
        
        hull = scipy_hull(points)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(points[:, 0], points[:, 1], 'o', color='#3b82f6', label='Internal Points')
        # Highlight points on the hull
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'r-', linewidth=2)
        ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'ro', label='Hull Vertices')
        
        # Annotate the "hidden" point (4,1)
        ax.annotate("Hidden Point (4,1)", (4.1, 1.1), color='gray', fontsize=9)
        
        ax.set_xlim(0, 8); ax.set_ylim(0, 6)
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        st.success("Note how (4,1) is inside the boundary. The Rubber Band doesn't touch it!")

    with tab_image:
        st.subheader("Object Enclosure Lab")
        st.sidebar.markdown("### 📥 Image Lab")
        local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="hull_local")
        binary_img = load_img_hull(local_up.read() if local_up else None)
        
        # Get coordinates of all foreground pixels
        coords = np.argwhere(binary_img)
        
        if len(coords) > 2:
            # SciPy hull works on coordinate sets
            hull_obj = scipy_hull(coords)
            
            # Create the hull mask for visualization
            from matplotlib.patches import Polygon
            fig_img, ax_img = plt.subplots()
            ax_img.imshow(binary_img, cmap='gray')
            
            # Get vertices in order
            hull_path = coords[hull_obj.vertices]
            polygon = Polygon(hull_path[:, [1, 0]], closed=True, fill=True, facecolor='red', alpha=0.3, label='Convex Hull')
            ax_img.add_patch(polygon)
            ax_img.set_title("Object with Enclosing Hull")
            ax_img.axis('off')
            st.pyplot(fig_img)
            
            st.info("The red area shows the 'rubber band' filling every concavity of the white object.")
        else:
            st.warning("Needs at least 3 pixels to form a hull.")

    with tab_metric:
        st.subheader("Solidity Dashboard")
        st.markdown("Solidity helps distinguish shapes based on their 'Fullness'.")
        
        if len(coords) > 2:
            hull_vol = hull_obj.volume # In 2D, volume is area
            obj_area = np.sum(binary_img)
            solidity = obj_area / hull_vol
            
            s_col1, s_col2 = st.columns(2)
            s_col1.metric("Object Area", f"{obj_area} px")
            s_col2.metric("Hull Area", f"{int(hull_vol)} px")
            
            st.progress(min(solidity, 1.0), text=f"Solidity Score: {solidity:.3f}")
            
            if solidity > 0.9:
                st.write("🟢 **Highly Convex**: This shape is very 'full' (like a circle or block).")
            elif solidity > 0.7:
                st.write("🟡 **Moderately Concave**: Some significant dents (like a crescent).")
            else:
                st.write("🔴 **Highly Concave**: Very deep dents or gaps (like a star or U-shape).")
        else:
            st.warning("Not enough pixels to calculate metrics.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Convex Hull**:
    - **Logic**: Smallest convex polygon containing all points/pixels.
    - **Analogy**: Rubber band tightening around nails.
    - **Metrics**: **Solidity** = $\text{Area}_{obj} / \text{Area}_{hull}$.
    - **Algorithms**: Graham Scan ($N \log N$), QuickHull, Jarvis's March.
    - **Usage**: Shape analysis, separating concave shapes from convex ones.
    """)
