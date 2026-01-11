# demos/spectrum_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from .dsp_utils import Wave, Spectrum

def run():
    st.subheader("Spectrum in DSP")

    # --- Theory Section (Expander) -------------------------------------------
    with st.expander("📝 Theory: Spectrum, DFT, and Filtering", expanded=False):
        st.markdown(
            r"""
            **A spectrum (plural: spectra)** describes a signal by showing what **frequencies** it contains.

            ### 1) What the spectrum represents
            - **Time Domain**: Signal value vs time (audio).
            - **Frequency Domain (Spectrum)**: Signal content vs frequency.
            - **Magnitude**: “How strong is this frequency?”
            - **Phase**: “How is this frequency shifted?”

            ### 2) Core nature and Calculation
            **Spectral Decomposition**: A signal can be decomposed into a sum of sinusoids.
            - **DFT (Discrete Fourier Transform)**: Converts signal samples $ \to $ spectrum.
            - **FFT (Fast Fourier Transform)**: A fast algorithm to compute the DFT.
            
            The DFT outputs complex numbers containing:
            - **Magnitude Spectrum** $|X[k]|$
            - **Phase Spectrum** $\angle X[k]$

            ### 3) Visualizing a Spectrum
            - **X-axis**: Frequency (Hz)
            - **Y-axis**: Amplitude/Magnitude
            - **Fundamental Frequency**: Lowest main frequency.
            - **Harmonics**: Integer multiples of the fundamental (2f, 3f...); these determine the "timbre" or shape.
            - **DC Component**: Frequency at 0 Hz (average value).

            ### 4) Comparison to `thinkdsp`
            - `signal.make_wave()` $\to$ **Wave** (Time domain)
            - `wave.make_spectrum()` $\to$ **Spectrum** (Frequency domain)
            - `spectrum.make_wave()` $\to$ **Wave** (reconstructed from frequency coeffs)
            
            **Filtering** happens in the frequency domain:
            - **Low-pass**: Remove high frequencies (smoothing).
            - **High-pass**: Remove low frequencies (sharpening).
            """
        )

    st.markdown("---")

    # --- Interactive Demo to Verify Logic ------------------------------------
    # "Full Practical / Mini Project" flow

    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Generate Signal")
        sig_type = st.selectbox("Signal Type", ["Square Wave", "Sawtooth Wave", "Sinusoid + Noise"])
        freq = st.slider("Fundamental Frequency (Hz)", 100, 1000, 220, step=10)
        duration = 1.0
        framerate = 10000

        # Generate Wave
        t = np.linspace(0, duration, int(framerate*duration), endpoint=False)
        
        if sig_type == "Square Wave":
            # Sign(sin)
            ys = 0.5 * np.sign(np.sin(2*np.pi*freq*t))
        elif sig_type == "Sawtooth Wave":
            # Linear rise then drop
            # easy approx using modulus
            # (t * freq) % 1  -> 0..1 saw
            ys = 2 * ((t * freq) % 1) - 1
            ys *= 0.5
        else:
            # Sinusoid + Noise
            ys = 0.5 * np.cos(2*np.pi*freq*t)
            ys += 0.2 * np.random.normal(size=len(t))

        wave = Wave(ys, t, framerate)
        
        st.markdown("---")
        st.write("### 3. Filter Spectrum")
        filter_type = st.radio("Filter Type", ["None", "Low Pass (cutoff)", "High Pass (cutoff)"])
        cutoff = st.slider("Cutoff Frequency (Hz)", 0, int(framerate/2), int(freq * 4))

    with col_viz:
        st.write("### 2. Time & Frequency Views")
        
        # A) Show Original Wave
        st.markdown("**Original Waveform** (Time Domain)")
        st.pyplot(wave.plot())
        st.audio(wave.get_audio_bytes(), format='audio/wav')

        # B) Compute Spectrum
        spectrum = wave.make_spectrum()
        
        # C) Apply Filter
        if filter_type == "Low Pass (cutoff)":
            spectrum.low_pass(cutoff)
            st.info(f"Applied Low Pass Filter at {cutoff} Hz")
        elif filter_type == "High Pass (cutoff)":
            spectrum.high_pass(cutoff)
            st.info(f"Applied High Pass Filter at {cutoff} Hz")

        # D) Show Spectrum
        st.markdown("**Spectrum** (Frequency Domain)")
        st.pyplot(spectrum.plot(high=freq*10)) # Zoom in to 10x fundamental usually enough
        
        # E) Inverse Transform (Reconstruct)
        st.markdown("**Reconstructed Wave** (Post-Filter)")
        
        recon_wave = spectrum.make_wave()
        st.pyplot(recon_wave.plot())
        st.audio(recon_wave.get_audio_bytes(), format='audio/wav')
        
        st.markdown(
            """
            **Observation**:
            - If you use a **Low Pass** filter on a Square/Sawtooth wave, you remove the "sharp corners" (high harmonics), making it look smoother/more sinusoidal.
            - If you use a **High Pass**, you remove the base shape and keep the "edges" or high-frequency noise.
            """
        )
