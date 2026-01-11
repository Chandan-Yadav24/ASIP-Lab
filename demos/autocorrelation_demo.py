# demos/autocorrelation_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Autocorrelation: Periodicity and Persistence")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: The Autocorrelation Function (ACF)", expanded=False):
        st.markdown(
            r"""
            **Autocorrelation** measures the similarity between a signal and its own delayed copy across a range of **lags**.
            
            ### 1) Mechanics: "Musical Echo"
            Imagine singing in a canyon. Autocorrelation is like recording your voice and playing it back with a shift. 
            When the shift match your singing loop exactly, the overlap "spikes."
            
            ### 2) The Dot Product Connection
            For zero-mean, normalized signals:
            $$\text{Autocorrelation} \approx \text{Signal} \cdot \text{Shifted Signal}$$
            It measures the "overlap" or linear similarity at each lag.
            
            ### 3) Key Applications
            - **Pitch Detection**: Find the period (time between spikes) to get fundamental frequency.
            - **Noise Identification**: 
                - **White Noise**: ACF drops to ~0 immediately (no memory).
                - **Pink Noise**: ACF persists over many lags (long-range dependence).
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Select Signal Type")
        mode = st.radio(
            "Signal Scenario", 
            ["Periodic Signal (Pitch)", "Uncorrelated Noise (White)", "Correlated Noise (Pink)"],
            key="acf_mode"
        )
        
        if mode == "Periodic Signal (Pitch)":
            freq = st.slider("Frequency (Hz)", 100, 1000, 440, key="acf_freq")
            noise_amp = st.slider("Add Noise", 0.0, 1.0, 0.2, key="acf_noise")
            sig = dsp_utils.Sinusoid(freq=freq) + dsp_utils.WhiteNoise(amp=noise_amp)
            desc = "Look for the spike! The lag of the first peak after zero is the period."
        elif mode == "Uncorrelated Noise (White)":
            sig = dsp_utils.UncorrelatedGaussianNoise(amp=1.0)
            desc = "Independence: The correlation drops to 0 immediately after lag 0."
        else: # Pink
            sig = dsp_utils.PinkNoise(amp=1.0, beta=1.0)
            desc = "Persistence: Correlation stays above zero for many lags (long memory)."
            
        wave = sig.make_wave(duration=0.1, framerate=11025)
        
        st.write("### 3. Listen")
        st.audio(wave.get_audio_bytes(), format='audio/wav')
        
        # Calculation
        ys = wave.ys
        ys_norm = ys - np.mean(ys)
        n = len(ys_norm)
        
        # Full ACF
        acf_full = np.correlate(ys_norm, ys_norm, mode='full')
        lags_full = np.arange(-n + 1, n)
        
        # Normalized and positive side only
        half_idx = len(lags_full) // 2
        lags = lags_full[half_idx:]
        acf = acf_full[half_idx:]
        acf = acf / acf[0] # Peak at lag 0 is 1.0
        
        # Limits for plot
        max_lag_view = st.slider("Max Lag to View", 20, 500, 200, key="acf_max_lag")

        st.info(f"**Insight**: {desc}")
        
    with col_viz:
        st.write("### 2. ACF Curve Visualizer")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot signal slice
        ax1.plot(wave.ts[:500], wave.ys[:500], color='tab:blue', alpha=0.7)
        ax1.set_title("Input Signal (Sample View)")
        ax1.set_xlabel("Time (s)")
        ax1.grid(alpha=0.3)
        
        # Plot ACF
        ax2.plot(lags[:max_lag_view], acf[:max_lag_view], color='tab:red', linewidth=2)
        ax2.axhline(0, color='black', alpha=0.5)
        ax2.set_title("Autocorrelation Function (ACF)")
        ax2.set_xlabel("Lag (samples)")
        ax2.set_ylabel("Correlation (ρ)")
        ax2.grid(alpha=0.3)
        
        # Mark peak if periodic
        if mode == "Periodic Signal (Pitch)":
            expected_period = int(wave.framerate / freq)
            if expected_period < max_lag_view:
                ax2.annotate(f'Peak at lag {expected_period}\n(Period)', 
                             xy=(expected_period, acf[expected_period]),
                             xytext=(expected_period + 20, acf[expected_period] + 0.2),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Results metrics
        if mode == "Periodic Signal (Pitch)":
            st.write(f"**Period Detection**: Theoretical period is **{expected_period}** samples.")
        elif mode == "Uncorrelated Noise (White)":
            st.write("**Rapid Decay**: Notice how ACF is nearly zero for any lag > 0.")
        else: # Pink
            st.write("**Slow Decay**: Correlation remains significant, proving long-range dependence.")
