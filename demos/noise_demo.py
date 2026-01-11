# demos/noise_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Uncorrelated Noise: Uniform vs Gaussian")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Uncorrelated Noise (UU vs UG)", expanded=False):
        st.markdown(
            r"""
            **Uncorrelated Noise**: Noise where each sample is independent of the others (no memory).

            ### 1) Two Main Types
            1. **Uncorrelated Uniform Noise (UU)**:
               - Values are drawn from a **Uniform Distribution** (all values in range roughly equally likely).
               - "Box" shaped histogram.
            2. **Uncorrelated Gaussian Noise (UG)**:
               - Values are drawn from a **Normal Distribution** (Bell Curve).
               - Clusters near the mean (0).
               - Often called "White Noise" in DSP (due to flat spectrum).

            ### 2) Key Characteristics
            - **Spectral Whiteness**: Both have equal power at all frequencies (flat spectrum).
            - **Sound**: Both sound like "static" or hiss. The difference is subtle to the ear but distinct in statistics.
            - **Covariance**: 0 (Uncorrelated).
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Generate Noise")
        noise_type = st.radio("Noise Type", ["Uncorrelated Uniform (UU)", "Uncorrelated Gaussian (UG)"])
        amp = st.slider("Amplitude", 0.1, 1.0, 0.5)
        duration = st.slider("Duration (s)", 0.5, 2.0, 1.0)
        
        # Generator
        if noise_type == "Uncorrelated Uniform (UU)":
            signal = dsp_utils.UncorrelatedUniformNoise(amp=amp)
        else:
            signal = dsp_utils.UncorrelatedGaussianNoise(amp=amp)  # approximate amp as sigma/std dev
        
        framerate = 11025
        wave = signal.make_wave(duration=duration, framerate=framerate)
        
        st.write("### 3. Listen")
        st.audio(wave.get_audio_bytes(), format='audio/wav')
        st.caption("Notice: They sound very similar (static) despite having different probability distributions.")

    with col_viz:
        st.write("### 2. Visualize Analysis")
        
        # A) Waveform
        st.markdown("**A) Waveform (Time Domain)**")
        # Plot a small segment so user can see "jaggedness"
        segment_duration = 0.05 # 50ms
        segment_len = int(segment_duration * framerate)
        fig_time, ax_time = plt.subplots(figsize=(8, 2))
        ax_time.plot(wave.ts[:segment_len], wave.ys[:segment_len], lw=1, alpha=0.8, color='tab:gray')
        ax_time.set_title(f"First {segment_duration*1000:.0f}ms Zoom")
        ax_time.grid(alpha=0.3)
        st.pyplot(fig_time)

        # B) Histogram (The Key Distinction)
        st.markdown("**B) Probability Distribution (Histogram)** - *The Key Difference*")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 3))
        ax_hist.hist(wave.ys, bins=50, color='tab:green', alpha=0.7, density=True)
        ax_hist.set_title("Histogram of Amplitude Values")
        ax_hist.set_xlabel("Amplitude")
        if noise_type == "Uncorrelated Uniform (UU)":
            ax_hist.set_ylabel("Probability (Uniform = Flat Top)")
        else:
            ax_hist.set_ylabel("Probability (Gaussian = Bell Curve)")
        ax_hist.grid(alpha=0.3)
        st.pyplot(fig_hist)

        # C) Spectrum
        st.markdown("**C) Spectrum (Frequency Domain)**")
        spectrum = wave.make_spectrum()
        st.pyplot(spectrum.plot())
        st.caption("Both look 'White' (Flat/Random) on average.")
