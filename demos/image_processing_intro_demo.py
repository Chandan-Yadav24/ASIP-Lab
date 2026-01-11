# demos/image_processing_intro_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def run():
    st.header("🖼️ Introduction to Image Processing")

    # --- Section: What is Image Processing? ---
    with st.expander("🤔 What is Image Processing?", expanded=True):
        st.markdown("""
        Digital image processing is the use of computer algorithms to perform image processing on digital images. 
        It allows a much wider range of algorithms to be applied to the input data and can avoid problems such as 
        the build-up of noise and signal distortion during processing.
        
        **Main Steps:**
        1. **Acquiring**: Capturing or scanning.
        2. **Processing**: Enhancing or filtering.
        3. **Analyzing**: Finding patterns or objects.
        4. **Interpreting**: Results for human or machine use.
        """)

    # --- Section: Analog vs Digital ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ Types")
        st.markdown("""
        - **Analog**: Works on photographs/print. Uses optical/chemical methods.
        - **Digital**: Works on computers. Uses math and code.
        """)
    with col2:
        st.subheader("💡 Why we need it?")
        st.markdown("""
        - **Human Perception**: Better visual quality.
        - **Machine Tasks**: Robotics/Auto-detection.
        - **Efficiency**: Storage and transmission.
        """)

    st.divider()

    # --- Section: Image as a Matrix ---
    st.subheader("🔢 Digital Image Representation")
    st.markdown(r"""
    A digital image is a 2D function $f(x, y)$, where $x$ and $y$ are spatial coordinates. 
    The value of $f$ is the **intensity** (brightness) at that point.
    """)

    tab1, tab2 = st.tabs(["🖤 Grayscale (8-bit)", "🌈 Color (RGB)"])

    with tab1:
        st.markdown("In a **Grayscale** image, each pixel is a single value from **0 (Black)** to **255 (White)**.")
        val = st.slider("Observe Intensity Value", 0, 255, 128)
        
        # Display the color block
        fig, ax = plt.subplots(figsize=(2, 2))
        patch = np.full((10, 10), val)
        ax.imshow(patch, cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        st.pyplot(fig)
        st.caption(f"Intensity Value: {val}")

        st.markdown("### Matrix View (Live)")
        uploaded_file = st.file_uploader("Upload an image to see its matrix", type=['png', 'jpg', 'jpeg'], key="intro_upload")
        
        if uploaded_file:
            img = Image.open(uploaded_file).convert('L')
            img_arr = np.array(img)
            
            st.write("Zooming into an 8x8 corner patch:")
            patch_8x8 = img_arr[:8, :8]
            st.table(patch_8x8)
        else:
            st.info("Upload an image to see the raw numbers behind the pixels!")

    with tab2:
        st.markdown("""
        **RGB (Red, Green, Blue)** images combine three primary colors to create millions of colors.
        Each channel typically uses 8 bits (0-255).
        """)
        
        r = st.slider("Red Channel", 0, 255, 255)
        g = st.slider("Green Channel", 0, 255, 0)
        b = st.slider("Blue Channel", 0, 255, 0)
        
        # Display combined color
        color_patch = np.zeros((10, 10, 3), dtype=np.uint8)
        color_patch[:, :, 0] = r
        color_patch[:, :, 1] = g
        color_patch[:, :, 2] = b
        
        fig_c, ax_c = plt.subplots(figsize=(2, 2))
        ax_c.imshow(color_patch)
        ax_c.axis('off')
        st.pyplot(fig_c)
        st.caption(f"RGB Combination: ({r}, {g}, {b})")

        st.markdown("### Channel Splitting")
        if uploaded_file:
            img_color = Image.open(uploaded_file).convert('RGB')
            arr_color = np.array(img_color)
            
            c_channels = st.columns(3)
            channels = ['Red', 'Green', 'Blue']
            for i, col in enumerate(c_channels):
                with col:
                    chan_arr = np.zeros_like(arr_color)
                    chan_arr[:,:,i] = arr_color[:,:,i]
                    st.image(chan_arr, caption=f"{channels[i]} Channel")
