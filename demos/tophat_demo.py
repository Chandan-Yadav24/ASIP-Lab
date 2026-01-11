# demos/tophat_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, color, util
from PIL import Image
import io

@st.cache_data
def load_img_tophat(file_data, test_case="Synthetic"):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
    else:
        if test_case == "Bright Details":
            # Dark background with small bright spots
            img_np = np.full((300, 300), 50, dtype=np.uint8)
            # Add some smooth background variation
            yy, xx = np.mgrid[:300, :300]
            img_np = (img_np + 20 * np.sin(xx/50) * np.cos(yy/50)).astype(np.uint8)
            # Add small bright spots
            for _ in range(15):
                ry, rxSize = np.random.randint(0, 290, 2)
                img_np[ry:ry+3, rxSize:rxSize+3] = 230
        else:
            # Grayscale background with dark cracks
            img_np = np.full((300, 300), 200, dtype=np.uint8)
            yy, xx = np.mgrid[:300, :300]
            # Add some smooth variation
            img_np = (img_np - 30 * np.cos(xx/60)).astype(np.uint8)
            # Add dark cracks/lines
            img_np[50:250, 150] = 50 # Vertical
            img_np[150, 50:250] = 50 # Horizontal
            # Add small dark spots
            for _ in range(10):
                ry, rxSize = np.random.randint(100, 200, 2)
                img_np[ry:ry+4, rxSize:rxSize+4] = 30
    return img_np

def run():
    st.header("🎩 Top-Hat Transforms: Detail Extractors")
    st.markdown("""
    Top-hat transforms are used to extract small bright (White Top-Hat) or dark (Black Top-Hat) features that are "sitting on" a non-uniform background.
    """)

    with st.expander("📚 Theory: Filtering the Background", expanded=False):
        st.markdown(r"""
        ### 1. White Top-Hat (WTH)
        **Formula**: $WTH(A) = A - (A \circ B)$
        - **Operation**: Original minus Opening.
        - **Result**: Small bright details that are smaller than the structuring element $B$.
        - **Use Case**: Enhancing bright spots on a gray/textured background.
        
        ### 2. Black Top-Hat (BTH)
        **Formula**: $BTH(A) = (A \bullet B) - A$
        - **Operation**: Closing minus Original.
        - **Result**: Small dark details (holes/cracks) smaller than the structuring element $B$.
        - **Use Case**: Highlighting dark pores or defects.
        
        ### 3. Key Insight
        Opening/Closing creates a "smoothed" version of the image that follows the background. Subtracting this from the original cancels out the background and leaves only the "peaks" (WTH) or "valleys" (BTH).
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    test_case = st.sidebar.selectbox("Test Case", ["Bright Details", "Dark Details"])
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="tophat_local")
    
    img = load_img_tophat(local_up.read() if local_up else None, test_case=test_case)
    
    st.sidebar.markdown("### 🧰 Structuring Element (SE)")
    se_shape = st.sidebar.selectbox("SE Shape", ["Disk", "Square", "Diamond"])
    se_size = st.sidebar.slider("SE Size (Radius/Size)", 1, 30, 5)
    
    if se_shape == "Disk":
        se = morphology.disk(se_size)
    elif se_shape == "Square":
        se = morphology.square(se_size * 2 + 1)
    else:
        se = morphology.diamond(se_size)

    tab_wth, tab_bth, tab_step = st.tabs(["✨ White Top-Hat", "🌑 Black Top-Hat", "🎞️ Step Analysis"])

    with tab_wth:
        st.subheader("Extracting Bright Peaks")
        st.markdown(r"$A - (A \circ B)$")
        
        wth = morphology.white_tophat(img, se)
        
        w1, w2 = st.columns(2)
        w1.image(img, caption="Original Grayscale", use_container_width=True)
        # Rescale for visibility
        wth_viz = util.img_as_ubyte(color.label2rgb(wth > 10, image=wth, bg_label=0)) if np.max(wth) > 0 else wth
        w2.image(wth, caption="White Top-Hat (Bright Details)", use_container_width=True, clamp=True)
        
        st.info("The smooth background variation is removed, leaving only the sharp bright spots.")

    with tab_bth:
        st.subheader("Extracting Dark Valleys")
        st.markdown(r"$(A \bullet B) - A$")
        
        bth = morphology.black_tophat(img, se)
        
        b1, b2 = st.columns(2)
        b1.image(img, caption="Original Grayscale", use_container_width=True)
        b2.image(bth, caption="Black Top-Hat (Dark Details)", use_container_width=True, clamp=True)
        
        st.success("The dark lines and spots are 'pulled out' from the gray background.")

    with tab_step:
        st.subheader("Visualizing the Background 'Envelope'")
        mode = st.radio("Choose Transform:", ["White (Original - Opening)", "Black (Closing - Original)"])
        
        if "White" in mode:
            smoothed = morphology.opening(img, se)
            result = img.astype(float) - smoothed.astype(float)
            label = "Opening (Smooth Background)"
        else:
            smoothed = morphology.closing(img, se)
            result = smoothed.astype(float) - img.astype(float)
            label = "Closing (Filled Background)"
            
        s1, s2, s3 = st.columns(3)
        s1.image(img, caption="1. Original", use_container_width=True)
        s2.image(smoothed, caption=f"2. {label}", use_container_width=True)
        s3.image(np.clip(result, 0, 255).astype(np.uint8), caption="3. Difference", use_container_width=True)
        
        st.markdown(f"**Insight**: Step 2 creates a version without the details. Subtracting it from Step 1 isolates them.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    - **White Top-Hat**: $A - (A \circ B)$. Highlights bright spots and lines.
    - **Black Top-Hat**: $(A \bullet B) - A$. Highlights dark cracks and spots.
    - **Goal**: Background subtraction and detail enhancement.
    - **Scaling**: The $B$ size determines what is considered "detail" vs "background".
    """)
