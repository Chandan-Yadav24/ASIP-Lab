# demos/overlapping_fields_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter

def run():
    st.header("🔗 Overlapping Fields with Image Processing")

    # --- Section: Theory ---
    with st.expander("🔬 The Processing Continuum", expanded=True):
        st.markdown("""
        Image processing doesn’t exist alone — it connects closely with **Computer Vision** and **Image Analysis**. 
        We can divide this continuum into three levels based on the complexity of the task and the nature of the output.
        """)
        
        st.info("💡 **Key Idea**: As the level increases, we move from manipulating 'Pixels' to understanding 'Concepts'.")

    # --- Interactive Levels Demo ---
    st.divider()
    st.subheader("Interactive Levels of Processing")
    
    uploaded_file = st.file_uploader("Upload Image to see levels", type=['jpg', 'jpeg', 'png'], key="overlap_upload")
    
    if uploaded_file:
        img_raw = Image.open(uploaded_file).convert('RGB')
    else:
        # Default placeholder: A simple geometric scene
        img_raw = Image.new('RGB', (300, 300), color=(200, 200, 200))
        # Draw a red circle and a blue square (conceptual)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_raw)
        draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0))
        draw.rectangle([180, 180, 280, 280], fill=(0, 0, 255), outline=(0, 0, 0))

    tab1, tab2, tab3 = st.tabs(["🧩 Low-Level", "🧭 Mid-Level", "🤖 High-Level"])

    with tab1:
        st.markdown("### 1. Low-Level Process")
        st.markdown("**Input: Image** $\\rightarrow$ **Output: Improved Image**")
        st.caption("Focuses on pixel-wise improvements like noise reduction or sharpening.")
        
        op = st.radio("Choose Operation", ["Normal", "Grayscale", "Auto-Contrast", "Sharpen"])
        
        if op == "Grayscale":
            res = img_raw.convert('L')
        elif op == "Auto-Contrast":
            res = ImageOps.autocontrast(img_raw)
        elif op == "Sharpen":
            res = img_raw.filter(ImageFilter.SHARPEN)
        else:
            res = img_raw
            
        st.image(res, caption="Output: Still an Image", width=400)

    with tab2:
        st.markdown("### 2. Mid-Level Process")
        st.markdown("**Input: Image** $\\rightarrow$ **Output: Attributes/Features**")
        st.caption("Extracts useful info like edges, contours, or object properties.")
        
        feat = st.radio("Choose Feature", ["Edge Detection", "Color Histogram", "Region Segmentation"])
        
        if feat == "Edge Detection":
            res_feat = img_raw.filter(ImageFilter.FIND_EDGES)
            st.image(res_feat, caption="Output: Edges (Structural Feature)", width=400)
        elif feat == "Color Histogram":
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ('red', 'green', 'blue')
            for i, color in enumerate(colors):
                hist = img_raw.histogram()[i*256:(i+1)*256]
                ax.plot(hist, color=color, alpha=0.7)
            ax.set_xlim([0, 256])
            ax.set_title("Color Distribution Attributes")
            st.pyplot(fig)
        else:
            # Simple thresholding for segmentation
            res_feat = img_raw.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
            st.image(res_feat, caption="Output: Segments (Binary Map)", width=400)

    with tab3:
        st.markdown("### 3. High-Level Process")
        st.markdown("**Input: Features** $\\rightarrow$ **Output: Understanding / Decisions**")
        st.caption("Interprets the scene and makes decisions based on extracted attributes.")
        
        st.warning("🤖 **Object Recognition Simulation**")
        
        # Mocking vision logic
        st.write("AI Analysis Result:")
        # For our default image, we can be specific
        if not uploaded_file:
            st.success("✅ Scene Understanding: Found 1 Red Circle and 1 Blue Square.")
            st.info("Decision: 'Move Robot to Square'")
        else:
            st.info("Scene Understanding: Interpreting complex visual context...")
            st.text("Detected Objects: [Person: 92%, Background: 100%]")
        
        # Concept visualization: Object detection box
        fig_ai, ax_ai = plt.subplots()
        ax_ai.imshow(img_raw)
        if not uploaded_file:
            import matplotlib.patches as patches
            rect = patches.Rectangle((50, 50), 100, 100, linewidth=2, edgecolor='r', facecolor='none')
            ax_ai.add_patch(rect)
            ax_ai.text(50, 45, "CIRCLE (Concept)", color='r')
        ax_ai.axis('off')
        st.pyplot(fig_ai)

    # --- Summary Table ---
    st.divider()
    st.subheader("📊 Hierarchy Summary")
    st.table({
        "Level": ["Low-Level", "Mid-Level", "High-Level"],
        "Input": ["Image", "Image", "Features"],
        "Output": ["Image (Enhanced)", "Features (Edges/Shapes)", "Understanding (Labels/Actions)"],
        "Analogy": ["Cleaning the lens", "Finding shapes", "Knowing what it is"]
    })
