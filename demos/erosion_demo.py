# demos/erosion_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import morphology, color, util
import io

@st.cache_data
def load_img_erosion(file_data, threshold=128):
    if file_data:
        img = Image.open(io.BytesIO(file_data)).convert('L')
        img_np = np.array(img)
        binary = img_np > threshold
    else:
        # Synthetic shapes
        img_np = np.zeros((400, 400), dtype=bool)
        img_np[50:150, 50:250] = True # Rect
        rr, cc = np.ogrid[:400, :400]
        mask = (rr - 250)**2 + (cc - 250)**2 <= 60**2
        img_np[mask] = True
        img_np[100:110, 200:350] = True # Bridge
        # Sparse Noise
        noise = np.random.rand(400, 400) > 0.995
        img_np = img_np | noise
        binary = img_np
    return binary

def get_se(shape, size):
    if shape == "Square":
        return morphology.square(size)
    elif shape == "Disk":
        return morphology.disk(size // 2 if size > 1 else 1)
    else: # Cross
        return morphology.star(size // 2 if size > 1 else 1)

def run():
    st.header("📉 Morphological Erosion: Shrinking & Filtering")
    st.markdown("""
    Erosion is a fundamental operation in Mathematical Morphology used to **shrink objects**, remove tiny details, and separate touching structures.
    """)

    with st.expander("📚 Theory: What is Erosion?", expanded=False):
        # ... (theory remains same)
        st.markdown(r"""
        ### 1. Intuition
        Think of sliding a template (Structuring Element) over an object. 
        - If the template fits **ENTIRELY** inside the object, we keep the center pixel.
        - If any part of the template touches the background, the center pixel is erased.
        
        ### 2. Binary Definition
        $$A \ominus B = \{z \mid B_z \subseteq A\}$$
        Where $A$ is the image and $B$ is the Structuring Element (SE).
        
        ### 3. Key Effects
        1. **Shrinks Objects**: Boundaries move inward.
        2. **Removes Noise**: Small white specks (smaller than SE) disappear.
        3. **Separates Touching Objects**: Narrow bridges are broken.
        4. **Boundary Extraction**: Outline = $A - (A \ominus B)$.
        
        ### 4. Grayscale Erosion
        For non-binary images, erosion acts as a **MIN Filter**. Every pixel is replaced by the lowest intensity in its neighborhood. This thins bright regions and pushes overall brightness down.
        """)

    st.sidebar.markdown("### 📥 Image Lab")
    local_up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="erosion_local")
    
    thresh_val = 128
    if local_up:
        thresh_val = st.sidebar.slider("Binary Threshold", 0, 255, 128)
        
    binary_img = load_img_erosion(local_up.read() if local_up else None, threshold=thresh_val)
    
    st.sidebar.markdown("### 🧰 Structuring Element (SE)")
    se_shape = st.sidebar.selectbox("Shape", ["Square", "Disk", "Cross"])
    se_size = st.sidebar.slider("Size", 1, 15, 3, step=2)
    se = get_se(se_shape, se_size)

    tab_binary, tab_boundary, tab_gray = st.tabs(["🔳 Binary Lab", "➰ Boundary Extraction", "🌫️ Grayscale (Min Filter)"])

    with tab_binary:
        st.subheader("Object Shrinking & Noise Removal")
        
        iterations = st.slider("Successive Erosions", 1, 5, 1)
        
        eroded = binary_img.copy()
        for _ in range(iterations):
            eroded = morphology.binary_erosion(eroded, se)
            
        col1, col2 = st.columns(2)
        # Convert booleans to uint8 * 255 for proper display
        col1.image(binary_img.astype(np.uint8) * 255, caption="Original Binary", use_container_width=True)
        col2.image(eroded.astype(np.uint8) * 255, caption=f"Eroded ({iterations} iter, size={se_size})", use_container_width=True)
        
        st.info("Notice how the small noise specks disappear first, and narrow bridges are broken.")

    with tab_boundary:
        st.subheader("Extracting the Outline")
        st.markdown(r"Boundary = $A - (A \ominus B)$")
        
        # We want a thin erosion for a thin boundary
        se_bound = morphology.square(3)
        eroded_b = morphology.binary_erosion(binary_img, se_bound)
        boundary = binary_img ^ eroded_b # XOR 
        
        b1, b2, b3 = st.columns(3)
        b1.image(binary_img.astype(np.uint8) * 255, caption="Original", use_container_width=True)
        b2.image(eroded_b.astype(np.uint8) * 255, caption="Eroded", use_container_width=True)
        b3.image(boundary.astype(np.uint8) * 255, caption="Extracted Boundary", use_container_width=True)
        
        st.success("By subtracting the 'shrunk' version from the original, only the boundary remains.")

    with tab_gray:
        st.subheader("Grayscale Erosion (Min Filter)")
        st.markdown("Bright regions shrink, dark regions expand.")
        
        # Load a grayscale version
        if local_up:
            gray_img = np.array(Image.open(io.BytesIO(local_up.getvalue())).convert('L'))
        else:
            # Use a noisy grayscale gradient
            gray_img = np.linspace(0, 255, 400).reshape(1, 400) * np.ones((400, 1))
            gray_img = gray_img + np.random.randint(0, 50, (400, 400))
            gray_img = np.clip(gray_img, 0, 255).astype(np.uint8)
            
        gray_eroded = morphology.erosion(gray_img, se)
        
        g1, g2 = st.columns(2)
        g1.image(gray_img, caption="Original Grayscale", use_container_width=True)
        g2.image(gray_eroded, caption="Grayscale Eroded (Minimum Filter)", use_container_width=True)
        
        st.caption("Grayscale erosion effectively 'thins' bright peaks and fills small dark valleys.")

    st.divider()
    st.markdown("### 📋 Exam Summary")
    st.info("""
    **Erosion** ($\ominus$): The 'Shrinker' of morphology.
    - **Binary**: $B$ must fit completely inside $A$.
    - **Results**: Removes noise specks, separates touching objects, thins boundaries.
    - **Boundary Formula**: $A \setminus (A \ominus B)$.
    - **Grayscale**: Equivalent to a nonlinear 'Minimum Filter'.
    """)
