# main.py
import streamlit as st
from demos import (
    periodic_demo, spectral_demo, signals_demo, wav_demo, spectrum_demo, 
    wave_object_demo, signal_object_demo, noise_demo, integrated_spectrum_demo, 
    brownian_demo, pink_noise_demo, gaussian_noise_demo, correlation_demo, 
    serial_correlation_demo, autocorrelation_demo, dot_product_demo, 
    freq_domain_demo, images_as_signals_demo, sampling_demo, dft_demo, 
    convolution_filtering_demo, smoothing_lpf_demo, sharpening_hpf_demo, 
    fft_mechanics_demo, image_processing_intro_demo, overlapping_fields_demo, image_types_formats_demo, intensity_transform_demo, 
    log_transform_demo, gamma_transform_demo, contrast_stretch_demo, thresholding_demo, 
    histogram_processing_demo, histogram_equalization_demo, smoothing_comparison_demo, sharpening_images_demo, 
    smoothing_comparison_demo, sharpening_images_demo, derivatives_gradients_demo, laplacian_demo, noise_gradient_demo, sobel_edge_demo, 
    canny_edge_demo, roberts_edge_demo, log_edge_demo, dog_filter_demo, gaussian_pyramid_demo, 
    laplacian_pyramid_demo, erosion_demo, dilation_demo, opening_closing_demo, hit_or_miss_demo, 
    skeleton_demo, convex_hull_demo, remove_objects_demo, tophat_demo, boundary_extraction_demo, 
    grayscale_morphology_demo, feature_extraction_intro_demo, boundary_feature_demo, pca_demo, harris_corner_demo,
    blob_detection_demo, hog_demo, sift_demo, haar_features_demo, segmentation_intro_demo, hough_transform_demo, hough_circle_demo,
    advanced_thresholding_demo, otsu_method_demo, edge_segmentation_demo, region_growing_demo, region_splitting_demo, region_merging_demo, watershed_demo, active_contours_demo, morphological_snakes_demo, grabcut_demo
)

st.set_page_config(page_title="ASIP Lab – Main Page", layout="wide")

def show_header():
    st.title("ASIP Lab – Streamlit Workspace")
    st.markdown(
        """
        Welcome to the **ASIP Lab** interactive workspace. 
        Select a Unit below to explore signal and image processing concepts.
        """
    )

def main():
    show_header()

    # Create Unit-based Tabs
    tab_unit1, tab_unit2, tab_unit3, tab_unit4 = st.tabs([
        "Unit 1: DSP Fundamentals", 
        "Unit 2: Digital Image Processing", 
        "Unit 3: Edge Detection & Segmentation",
        "Unit 4: Feature Extraction & Matching"
    ])

    with tab_unit1:
        st.subheader("Signal Processing Practicals")
        
        # Selection for Unit 1
        unit1_demos = {
            "1. Periodic Signal Demo": periodic_demo,
            "2. Spectral Decomposition Demo": spectral_demo,
            "3. Signals (1D & 2D) Demo": signals_demo,
            "4. Reading and Writing WAV Files": wav_demo,
            "5. Spectrum & Filtering Demo": spectrum_demo,
            "6. Wave Objects In-Depth": wave_object_demo,
            "7. Signal Objects & Hierarchy": signal_object_demo,
            "8. Uncorrelated Noise (UU vs UG)": noise_demo,
            "9. Integrated Spectrum (Noise Analysis)": integrated_spectrum_demo,
            "10. Brownian Noise Deep Dive": brownian_demo,
            "11. Pink Noise (1/f) Analysis": pink_noise_demo,
            "12. Gaussian Noise (Image Processing Interface)": gaussian_noise_demo,
            "13. Correlation (Similarity & Delays)": correlation_demo,
            "14. Serial Correlation (Signal Memory)": serial_correlation_demo,
            "15. Autocorrelation (Pitch & Noise Analysis)": autocorrelation_demo,
            "16. Correlation as a Dot Product": dot_product_demo,
            "17. Frequency-Domain Operations (Overview)": freq_domain_demo,
        }
        
        selected_u1 = st.selectbox("Select Practical", options=list(unit1_demos.keys()), key="u1_select")
        st.divider()
        unit1_demos[selected_u1].run()

    with tab_unit2:
        st.subheader("Image Processing Practicals")
        
        # Selection for Unit 2
        unit2_demos = {
            "0. What is Image Processing?": image_processing_intro_demo,
            "0.1 Overlapping Fields (Theory Level)": overlapping_fields_demo,
            "0.2 Image Types & File Formats": image_types_formats_demo,
            "0.3 Intensity Transformations": intensity_transform_demo,
            "0.4 Log Transformation In-Depth": log_transform_demo,
            "0.5 Power-law (Gamma) Transform": gamma_transform_demo,
            "0.6 Contrast Stretching": contrast_stretch_demo,
            "0.7 Thresholding & Segmentation": thresholding_demo,
            "0.8 Histogram Processing (Overview)": histogram_processing_demo,
            "0.9 Histogram Equalization (HE)": histogram_equalization_demo,
            "1.0 Linear and Non-linear Smoothing": smoothing_comparison_demo,
            "1.1 Sharpening of Images": sharpening_images_demo,
            "1.2 Derivatives and Gradients": derivatives_gradients_demo,
            "1.3 Laplacian in Image Processing": laplacian_demo,
            "1.4 Effect of Noise on Gradients": noise_gradient_demo,
            "18. Images as Signals (2D Representation)": images_as_signals_demo,
            "19. Sampling & Nyquist (Time/Frequency Link)": sampling_demo,
            "20. The Discrete Fourier Transform (DFT)": dft_demo,
            "21. Convolution & Spectral Filtering": convolution_filtering_demo,
            "22. Smoothing & Noise Reduction (LPF)": smoothing_lpf_demo,
            "23. Sharpening & High-Pass Filters (HPF)": sharpening_hpf_demo,
            "24. FFT Mechanics & Efficiency": fft_mechanics_demo,
        }
        
        selected_u2 = st.selectbox("Select Practical", options=list(unit2_demos.keys()), key="u2_select")
        st.divider()
        unit2_demos[selected_u2].run()

    with tab_unit3:
        st.subheader("Edge Detection & Segmentation")
        
        unit3_demos = {
            "3.1 Sobel Operator (Standard Edge Detector)": sobel_edge_demo,
            "3.2 Canny Edge Detector (Optimal & Thin Edges)": canny_edge_demo,
            "3.3 Roberts Cross Operator (Diagonal Specialist)": roberts_edge_demo,
            "3.4 LoG / Mexican Hat (Zero Crossings)": log_edge_demo,
            "3.5 DoG (Band-Pass & SIFT)": dog_filter_demo,
            "3.6 Gaussian Pyramid (Multi-Scale)": gaussian_pyramid_demo,
            "3.7 Laplacian Pyramid (Detail Reconstruction)": laplacian_pyramid_demo,
            "3.8 Morphological Erosion (Shrinking)": erosion_demo,
            "3.9 Morphological Dilation (Growth)": dilation_demo,
            "3.10 Morphological Opening & Closing (Clean & Repair)": opening_closing_demo,
            "3.11 Hit-or-Miss Transformation (Pattern Detector)": hit_or_miss_demo,
            "3.12 Skeletonizing (Thinning)": skeleton_demo,
            "3.13 Convex Hull (Rubber Band Enclosure)": convex_hull_demo,
            "3.14 Removing Small Objects (Cleanup)": remove_objects_demo,
            "3.15 White and Black Top-Hat Transforms (Detail Extraction)": tophat_demo,
            "3.16 Extracting the Boundary (Outline)": boundary_extraction_demo,
            "3.17 Grayscale Morphology (Intensity Ops)": grayscale_morphology_demo,
        }
        
        selected_u3 = st.selectbox("Select Practical", options=list(unit3_demos.keys()), key="u3_select")
        st.divider()
        unit3_demos[selected_u3].run()

    with tab_unit4:
        st.subheader("Feature Extraction & Matching")
        
        unit4_demos = {
            "4.1 Intro to Feature Extraction (Mid-level)": feature_extraction_intro_demo,
            "4.2 Boundary Processing & Feature Description": boundary_feature_demo,
            "4.3 Principal Component Analysis (PCA)": pca_demo,
            "4.3.1 Hough Transform for Line Detection": hough_transform_demo,
            "4.3.2 Hough Transform for Circle Detection": hough_circle_demo,
            "4.4 Harris Corner Detector (Interest Points)": harris_corner_demo,
            "4.5 Blob Detection (LoG, DoG, DoH)": blob_detection_demo,
            "4.6 Histogram of Oriented Gradients (HOG)": hog_demo,
            "4.7 Scale-Invariant Feature Transform (SIFT)": sift_demo,
            "4.8 Haar-like Features (Viola-Jones)": haar_features_demo,
            "4.9 Image Segmentation Intro (Hough, Otsu, Watershed)": segmentation_intro_demo,
            "4.10 Advanced Thresholding Operations": advanced_thresholding_demo,
            "4.11 Otsu's Method (Automatic Global Thresholding)": otsu_method_demo,
            "4.12 Edge-based Segmentation (Pipeline)": edge_segmentation_demo,
            "4.13 Region Growing Segmentation": region_growing_demo,
            "4.14 Region Splitting (Top-down)": region_splitting_demo,
            "4.15 Region Merging (Image Segmentation)": region_merging_demo,
            "4.16 Watershed Algorithm (Morphological Segmentation)": watershed_demo,
            "4.17 Active Contours (Snakes)": active_contours_demo,
            "4.18 Morphological Snakes": morphological_snakes_demo,
            "4.19 GrabCut Algorithm": grabcut_demo,
        }
        
        selected_u4 = st.selectbox("Select Practical", options=list(unit4_demos.keys()), key="u4_select")
        st.divider()
        selected_u4_val = unit4_demos[selected_u4]
        selected_u4_val.run()

    st.markdown("---")
    st.subheader("Upcoming Modules")
    st.info("- SIFT/SURF Matching, Object Tracking, Stereo Vision")

if __name__ == "__main__":
    main()
