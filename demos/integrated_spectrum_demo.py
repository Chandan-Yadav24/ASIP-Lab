# demos/integrated_spectrum_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Integrated Spectrum: Identifying Noise Types")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Integrated Spectrum Analysis", expanded=False):
        st.markdown(
            r"""
            **Integrated Spectrum**: A cumulative view of power distribution across frequencies.
            
            ### 1) Why Use It?
            - Power spectra of noise are **jagged** and hard to interpret.
            - Integrated spectrum **smooths** the view by showing cumulative power.
            - Makes it easy to identify **white** vs **colored** noise.
            
            ### 2) How It's Computed
            1. Start with power spectrum: $P[f]$
            2. Cumulative sum: $CS[k] = \sum_{i=0}^{k} P[i]$
            3. Normalize: $CS_{norm}[k] = CS[k] / CS[last]$ (so it ends at 1)
            
            ### 3) Interpretation
            - **Straight Line** (diagonal): White noise (equal power at all frequencies)
            - **Curved Up** (convex): More power at low frequencies (Pink/Brownian noise)
            - **Curved Down** (concave): More power at high frequencies (rare)
            
            ### 4) Analogy: Rain Gauge
            - Power spectrum = rainfall per minute (jagged)
            - Integrated spectrum = total collected water (smooth)
            - Straight line = constant rainfall rate
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Generate Noise")
        
        noise_type = st.selectbox(
            "Noise Type",
            ["White Noise (Uncorrelated)", "Pink Noise (1/f)", "Brownian Noise (Red)"]
        )
        
        duration = st.slider("Duration (s)", 0.5, 2.0, 1.0, key="integrated_spectrum_duration")
        framerate = 11025
        
        # Create signal
        if noise_type == "White Noise (Uncorrelated)":
            signal = dsp_utils.UncorrelatedGaussianNoise(amp=0.5)
            expected_shape = "Straight diagonal line"
        elif noise_type == "Pink Noise (1/f)":
            signal = dsp_utils.PinkNoise(amp=0.5, beta=1.0)
            expected_shape = "Curved upward (more low-freq power)"
        else:  # Brownian
            signal = dsp_utils.BrownianNoise(amp=0.5)
            expected_shape = "Strongly curved upward"
        
        wave = signal.make_wave(duration=duration, framerate=framerate)
        spectrum = wave.make_spectrum()
        integrated = spectrum.make_integrated_spectrum()
        
        st.info(f"**Expected Shape**: {expected_shape}")
        
        st.write("### 3. Listen")
        st.audio(wave.get_audio_bytes(), format='audio/wav')

    with col_viz:
        st.write("### 2. Compare Spectra")
        
        # A) Power Spectrum
        st.markdown("**A) Power Spectrum** (Jagged/Noisy)")
        st.pyplot(spectrum.plot())
        
        # B) Integrated Spectrum
        st.markdown("**B) Integrated Spectrum** (Smooth Cumulative)")
        st.pyplot(integrated.plot())
        
        # Interpretation
        st.markdown("---")
        st.markdown("### 🔍 Interpretation Guide")
        
        if noise_type == "White Noise (Uncorrelated)":
            st.success(
                "✅ **White Noise**: The integrated spectrum should follow the diagonal reference line closely. "
                "This means power is distributed evenly across all frequencies."
            )
        elif noise_type == "Pink Noise (1/f)":
            st.info(
                "🎨 **Pink Noise**: The curve rises faster at low frequencies, then slows down. "
                "This indicates more energy in bass/low frequencies (common in natural sounds)."
            )
        else:
            st.warning(
                "🟤 **Brownian Noise**: The curve rises very steeply at low frequencies. "
                "Almost all power is concentrated in the lowest frequencies (like a random walk)."
            )
