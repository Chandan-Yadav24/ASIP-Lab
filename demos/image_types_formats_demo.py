# demos/image_types_formats_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def run():
    st.header("🖼️ Image Types & File Formats")

    # --- Section: Digital Image Types ---
    st.subheader("1. Types of Digital Images")
    st.markdown("""
    Digital images are representations of light and color as numbers. 
    The complexity ranges from simple 1-bit binary to multi-band scientific data.
    """)

    tabs_types = st.tabs(["🩶 Binary", "🌫️ Grayscale", "🌈 Color", "🌍 Multispectral", "🔄 Type Conversion", "💾 Disk Format Converter"])

    # Shared User Image Upload
    st.sidebar.markdown("### 📥 Global Image Lab")
    uploaded_file = st.sidebar.file_uploader("Upload Image for Labs", type=['png', 'jpg', 'jpeg'], key="types_upload")
    
    if uploaded_file:
        img_raw = Image.open(uploaded_file)
        img_rgb = np.array(img_raw.convert('RGB'))
        img_gray = np.array(img_raw.convert('L'))
    else:
        # Placeholder image: A gradient to show transitions well
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        X, Y = np.meshgrid(x, y)
        img_gray = (X * 255).astype(np.uint8)
        # Create a colorful gradient for RGB placeholder
        img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        img_rgb[:, :, 0] = (X * 255).astype(np.uint8)
        img_rgb[:, :, 1] = (Y * 255).astype(np.uint8)
        img_rgb[:, :, 2] = ((1-X) * 255).astype(np.uint8)

    with tabs_types[0]:
        st.markdown("### (i) Binary Images (Black and White)")
        st.markdown("""
        **Characteristics:**
        - **Values:** 0 (Black) or 1 (White).
        - **Depth:** 1 bit per pixel ($2^1 = 2$ levels).
        - **Use Case:** Barcodes, signatures, document scanning.
        """)
        st.info("💡 Binary images are the most memory-efficient but lose all texture and shading.")

    with tabs_types[1]:
        st.markdown("### (ii) Grayscale Images")
        st.markdown("""
        **Characteristics:**
        - **Values:** 0 (Black) to 255 (White) for 8-bit.
        - **Depth:** Typically 8 bits per pixel ($2^8 = 256$ levels).
        - **Concept:** Represents 'Intensity' or 'Luminance' only.
        """)
        st.image(img_gray, caption="Grayscale Representation (8-bit)", width=400)

    with tabs_types[2]:
        st.markdown("### (iii) Color Images (RGB)")
        st.markdown("""
        **Characteristics:**
        - **Model:** Red, Green, Blue (Additive color model).
        - **Depth:** 24 bits per pixel ($8 \cdot 3$ channels).
        - **Range:** Over 16.7 million possible colors.
        """)
        
        c_channels = st.columns(3)
        channels = ['Red', 'Green', 'Blue']
        for i, col in enumerate(c_channels):
            with col:
                chan_data = np.zeros_like(img_rgb)
                chan_data[:, :, i] = img_rgb[:, :, i]
                st.image(chan_data, caption=f"{channels[i]} Channel")

    with tabs_types[3]:
        st.markdown("### (iv) Multispectral Images")
        st.markdown("""
        **Beyond Human Vision:**
        Humans see RGB, but sensors can capture **Infrared (Heat)**, **Ultraviolet**, and more.
        - **Applications:** Agriculture (crop health), Satellite mapping, Mineralogy.
        """)
        
        st.warning("🌍 **NASA Simulation**: Visualizing invisible bands.")
        bands = st.multiselect("Select Bands to View", 
                              ["Visible Red", "Visible Green", "Visible Blue", "Infrared (Thermal Sim)", "Ultraviolet (Chemical Sim)"], 
                              default=["Visible Red", "Visible Green", "Visible Blue"])
        
        fig_m, axes = plt.subplots(1, len(bands), figsize=(15, 3))
        if len(bands) == 1: axes = [axes]
        
        for i, band in enumerate(bands):
            if "Visible" in band: cmap = 'gray'
            elif "Infrared" in band: cmap = 'hot'
            else: cmap = 'magma'
            
            axes[i].imshow(img_gray, cmap=cmap)
            axes[i].set_title(band)
            axes[i].axis('off')
        st.pyplot(fig_m)

    with tabs_types[4]:
        st.markdown("### 🔄 Explicit Type Conversion")
        st.markdown("Transform the image data step-by-step and observe the 'Information Cost'.")
        
        conv_step = st.radio("Conversion Step", ["1. RGB to Grayscale", "2. Grayscale to Binary"], horizontal=True)
        
        if conv_step == "1. RGB to Grayscale":
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption="Source: 24-bit RGB", use_container_width=True)
            
            with col2:
                algo = st.selectbox("Conversion Algorithm", 
                                   ["Average ( (R+G+B)/3 )", "Luminance ( 0.299R + 0.587G + 0.114B )"])
                
                if "Average" in algo:
                    res_gray = np.mean(img_rgb, axis=2).astype(np.uint8)
                else:
                    # Specific weights for human eye perception
                    res_gray = (0.299*img_rgb[:,:,0] + 0.587*img_rgb[:,:,1] + 0.114*img_rgb[:,:,2]).astype(np.uint8)
                
                st.image(res_gray, caption=f"Result: 8-bit Grayscale ({algo.split(' ')[0]})", use_container_width=True)
            
            st.metric("Information Reduction", "66.6%", delta="-16 bits per pixel")
            st.caption("We went from 3 numbers per pixel to just 1.")

        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_gray, caption="Source: 8-bit Grayscale", use_container_width=True)
            
            with col2:
                thresh = st.slider("Threshold (0-255)", 0, 255, 128)
                res_bin = (img_gray > thresh).astype(np.uint8) * 255
                st.image(res_bin, caption="Result: 1-bit Binary", use_container_width=True)
            
            st.metric("Information Reduction", "87.5%", delta="-7 bits per pixel")
            st.caption("We went from 256 possible shades to just 2 (On/Off).")

    with tabs_types[5]:
        st.markdown("### 💾 Disk Format Converter")
        st.markdown("Convert your uploaded or sample image into common file formats and compare disk usage.")
        
        # Local Uploader for this tab
        local_file = st.file_uploader("📤 Convert Your Image (Overrides Sidebar)", type=['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'], key="local_conv_upload")
        
        # Determine source
        active_file = local_file if local_file else uploaded_file
        
        if active_file:
            pil_source = Image.open(active_file)
            source_name = getattr(active_file, 'name', "Uploaded Image")
            source_size = active_file.size / 1024
            source_fmt = pil_source.format
            # Force RGB for consistent conversion
            img_to_conv = np.array(pil_source.convert('RGB'))
        else:
            img_to_conv = img_rgb
            source_name = "Sample Gradient"
            source_size = 256.0
            source_fmt = "Generated"

        st.info(f"📁 **Source Info**: `{source_name}` | **Format**: {source_fmt} | **Size**: {source_size:.1f} KB")

        target_fmt = st.selectbox("Select Target Format", ["JPEG", "PNG", "GIF", "TIFF"])
        
        # Prepare for conversion
        buf_out = io.BytesIO()
        pil_img_out = Image.fromarray(img_to_conv)
        
        if target_fmt == "JPEG":
            q_conv = st.slider("Conversion Quality", 1, 100, 85, key="conv_q_enhance")
            pil_img_out.save(buf_out, format="JPEG", quality=q_conv)
            ext = ".jpg"
        elif target_fmt == "PNG":
            pil_img_out.save(buf_out, format="PNG", optimize=True)
            ext = ".png"
        elif target_fmt == "GIF":
            pil_img_out.convert("P", palette=Image.ADAPTIVE).save(buf_out, format="GIF")
            ext = ".gif"
        else: # TIFF
            pil_img_out.save(buf_out, format="TIFF")
            ext = ".tiff"
            
        new_size = buf_out.tell() / 1024
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(img_to_conv, caption="Source Preview", use_container_width=True)
            st.metric("Original Size", f"{source_size:.1f} KB")
        with c2:
            st.image(buf_out, caption=f"Converted Result ({target_fmt})", use_container_width=True)
            st.metric("New Size", f"{new_size:.1f} KB", delta=f"{(new_size - source_size):.1f} KB")
            
        st.download_button(
            label=f"⬇️ Download Converted {target_fmt}",
            data=buf_out.getvalue(),
            file_name=f"asip_converted_{target_fmt.lower()}{ext}",
            mime=f"image/{target_fmt.lower()}",
            use_container_width=True
        )

    st.divider()

    # --- Section: File Formats ---
    st.subheader("2. Image File Formats")
    st.markdown("How is this pixel data stored on your disk?")
    
    tab_gif, tab_jpg, tab_png, tab_tiff = st.tabs(["📄 GIF", "📸 JPEG", "🎨 PNG", "📠 TIFF"])

    with tab_gif:
        st.markdown("""
        ### GIF (Graphics Interchange Format)
        - **Logic:** Indexed color (max 256 colors).
        - **Compression:** Lossless (LZW).
        - **Bonus:** Supports transparency and simple animation.
        """)

    with tab_jpg:
        st.markdown("""
        ### JPEG (Joint Photographic Experts Group)
        - **Logic:** Uses Discrete Cosine Transform (DCT).
        - **Compression:** **Lossy** (Removes high-frequency detail human eyes don't notice).
        - **Strength:** Excellent for real photographs.
        """)
        st.info("🎞️ **Interactive: Quality vs. Artifacts**")
        q = st.slider("JPEG Quality Factor", 1, 100, 75, key="jpg_q_new")
        
        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="JPEG", quality=q)
        size_kb = buf.tell() / 1024
        st.image(Image.open(buf), caption=f"Quality: {q}% | Size: {size_kb:.1f} KB")

    with tab_png:
        st.markdown("""
        ### PNG (Portable Network Graphics)
        - **Logic:** Predictive filtering and Deflate compression.
        - **Compression:** Lossless (True quality).
        - **Bonus:** Alpha transparency (smooth fades into backgrounds).
        """)

    with tab_tiff:
        st.markdown("""
        ### TIFF (Tagged Image File Format)
        - **Logic:** High flexibility, can be uncompressed.
        - **Best For:** Archiving and professional printing where no loss is tolerated.
        """)

    # --- Summary Table ---
    st.divider()
    st.subheader("📊 Comparison Summary")
    st.table({
        "Feature": ["Bits/Pixel", "Compression", "Transparency", "Best For"],
        "Binary": ["1 bit", "None/RLE", "No", "Barcodes"],
        "Grayscale": ["8 bits", "Various", "No", "X-Rays"],
        "Color (RGB)": ["24 bits", "JPEG/PNG", "Yes (PNG/GIF)", "Photography"],
        "Multispectral": ["Variable", "Lossless", "No", "Satellites"]
    })
