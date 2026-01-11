# demos/freq_domain_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Frequency-Domain Operations: The FFT Workflow")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Working with Spices (Frequencies)", expanded=False):
        st.markdown(
            r"""
            **Frequency-domain operations** allow us to modify a signal by adjusting its spectrum instead of its raw values.
            
            ### 1) The 3-Step Workflow
            1. **Transform**: Convert signal to frequency domain (FFT).
            2. **Process**: Modify the frequency components (Filters).
            3. **Inverse Transform**: Convert back to time domain (IFFT).
            
            ### 2) The Gourmet Soup Analogy
            - **Time Domain**: The soup is mixed; hard to change one flavor.
            - **Frequency Domain**: You separate the soup into ingredients (salt, pepper, herbs), adjust individually, then mix back.
            
            ### 3) Convolution Theorem
            Convolution in time domain is equivalent to **multiplication** in frequency domain. 
            This makes large filters much faster to compute using the FFT.
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Design Signal")
        
        sig_type = st.radio("Base Signal", ["Square Wave (Rich Harmonics)", "Noisy Sine Wave"], key="fd_sig_type")
        
        if sig_type == "Square Wave (Rich Harmonics)":
            sig = dsp_utils.Sinusoid(freq=200) # We'll manually make square if needed, or just rich
            # dsp_utils doesn't have Square, let's use a combination
            sig = dsp_utils.Sinusoid(freq=200, amp=0.5) + dsp_utils.Sinusoid(freq=600, amp=0.2) + dsp_utils.Sinusoid(freq=1000, amp=0.1)
        else:
            sig = dsp_utils.Sinusoid(freq=440, amp=0.5) + dsp_utils.WhiteNoise(amp=0.3)
            
        wave = sig.make_wave(duration=0.5, framerate=11025)
        
        st.write("### 2. Apply Filter")
        filter_type = st.selectbox("Operation", ["None (Original)", "Low-pass (Smooth)", "High-pass (Sharpen)"], key="fd_filter_type")
        cutoff = st.slider("Cutoff Frequency (Hz)", 100, 5000, 1000, key="fd_cutoff")
        
        # Process in Frequency Domain
        spectrum = wave.make_spectrum()
        
        # Create filter mask
        if filter_type == "Low-pass (Smooth)":
            mask = spectrum.fs <= cutoff
            desc = "Removing high frequencies (the hiss/sharpness)."
        elif filter_type == "High-pass (Sharpen)":
            mask = spectrum.fs >= cutoff
            desc = "Removing low frequencies (the bass/thump)."
        else:
            mask = np.ones_like(spectrum.fs, dtype=bool)
            desc = "No modification."
            
        # Modify spectrum
        filtered_spectrum = dsp_utils.Spectrum(spectrum.hs * mask, spectrum.fs, spectrum.framerate)
        filtered_wave = filtered_spectrum.make_wave()
        
        st.write("### 3. Listen Comparison")
        st.audio(wave.get_audio_bytes(), format='audio/wav')
        st.caption("Original")
        
        st.audio(filtered_wave.get_audio_bytes(), format='audio/wav')
        st.caption(f"Processed ({filter_type})")
        
        st.info(f"**Status**: {desc}")

    with col_viz:
        st.write("### Analysis: Spectrum Processing")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
        
        # 1. Original Spectrum
        ax1.plot(spectrum.fs, np.abs(spectrum.hs), color='tab:gray', alpha=0.5, label='Original')
        ax1.set_title("Input Spectrum (Magnitude)")
        ax1.set_xlim(0, 5000)
        ax1.grid(alpha=0.2)
        
        # 2. Filter Mask Visualization
        m_vals = np.zeros_like(spectrum.fs)
        if filter_type == "Low-pass (Smooth)":
            m_vals[spectrum.fs <= cutoff] = 1
            ax2.fill_between(spectrum.fs, m_vals, color='tab:green', alpha=0.3, label='Pass-band')
        elif filter_type == "High-pass (Sharpen)":
            m_vals[spectrum.fs >= cutoff] = 1
            ax2.fill_between(spectrum.fs, m_vals, color='tab:blue', alpha=0.3, label='Pass-band')
        else:
            ax2.text(0.5, 0.5, "No Filter Active", ha='center', va='center')
            
        ax2.set_title(f"Spectral Filter Mask (Cutoff: {cutoff} Hz)")
        ax2.set_xlim(0, 5000)
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(alpha=0.1)
        
        # 3. Resulting Spectrum
        ax3.plot(filtered_spectrum.fs, np.abs(filtered_spectrum.hs), color='tab:red', linewidth=1.5)
        ax3.set_title("Modified Spectrum (Multiplication Result)")
        ax3.set_xlim(0, 5000)
        ax3.set_xlabel("Frequency (Hz)")
        ax3.grid(alpha=0.2)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Waveform comparison at the bottom
        st.markdown("**Resulting Waveform (Inverse FFT)**")
        fig_w, ax_w = plt.subplots(figsize=(10, 3))
        ax_w.plot(wave.ts[:500], wave.ys[:500], label='Original', alpha=0.4, color='gray')
        ax_w.plot(filtered_wave.ts[:500], filtered_wave.ys[:500], label='Filtered', color='tab:red')
        ax_w.set_title("Time Domain Comparison (Original vs Filtered)")
        ax_w.legend()
        ax_w.grid(alpha=0.2)
        st.pyplot(fig_w)
