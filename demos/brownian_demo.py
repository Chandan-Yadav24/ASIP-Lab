# demos/brownian_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Brownian Noise: The Random Walk")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Brownian (Red/Brown) Noise", expanded=False):
        st.markdown(
            r"""
            **Brownian Noise**: A correlated noise signal that behaves like a random walk.
            
            ### 1) Core Concept
            Each sample depends on the previous one:
            $$x[n] = x[n-1] + \text{random step}$$
            
            This creates a **wandering** pattern rather than random jumps.
            
            ### 2) Statistical Characteristics
            - **Highly Correlated**: Serial correlation ≈ 1 (often > 0.999)
            - **Memory**: Current value strongly depends on previous value
            - **Smooth Wandering**: Tends to stay high when high, low when low
            
            ### 3) Spectral Properties
            - **Power-Frequency Relationship**: $P(f) = K/f^2$
            - **Low-Frequency Dominance**: Most energy at bass frequencies
            - **"Red" Noise**: Named for concentration at low-frequency end (like red light)
            
            ### 4) Generation Method
            1. Generate white noise (random steps)
            2. Cumulative sum: `np.cumsum(steps)`
            3. Normalize amplitude
            
            ### 5) Analogy: Drunken Hiker
            - **White Noise**: Teleporting randomly each second
            - **Brownian Noise**: Taking small random steps, so position depends on last position
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Configuration")
        
        show_comparison = st.checkbox("Compare with White Noise", value=False, key="brownian_compare")
        duration = st.slider("Duration (s)", 0.5, 3.0, 1.5, key="brownian_duration")
        framerate = 11025
        
        # Generate Brownian noise
        brownian_signal = dsp_utils.BrownianNoise(amp=0.5)
        brownian_wave = brownian_signal.make_wave(duration=duration, framerate=framerate)
        
        # Generate White noise for comparison if needed
        if show_comparison:
            white_signal = dsp_utils.UncorrelatedGaussianNoise(amp=0.5)
            white_wave = white_signal.make_wave(duration=duration, framerate=framerate)
        
        st.write("### 3. Listen")
        st.audio(brownian_wave.get_audio_bytes(), format='audio/wav')
        st.caption("Notice the deep, rumbling sound")

    with col_viz:
        st.write("### 2. Analysis")
        
        # A) Waveform Comparison
        st.markdown("**A) Waveform: Random Walk Behavior**")
        
        if show_comparison:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            
            # Brownian
            segment = min(2000, len(brownian_wave.ys))
            ax1.plot(brownian_wave.ts[:segment], brownian_wave.ys[:segment], 
                    color='tab:brown', linewidth=1, alpha=0.8)
            ax1.set_title("Brownian Noise (Smooth Wandering)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(alpha=0.3)
            
            # White
            ax2.plot(white_wave.ts[:segment], white_wave.ys[:segment], 
                    color='tab:gray', linewidth=1, alpha=0.8)
            ax2.set_title("White Noise (Jagged/Random)")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Amplitude")
            ax2.grid(alpha=0.3)
            
            fig.tight_layout()
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 3))
            segment = min(2000, len(brownian_wave.ys))
            ax.plot(brownian_wave.ts[:segment], brownian_wave.ys[:segment], 
                   color='tab:brown', linewidth=1.5, alpha=0.8)
            ax.set_title("Brownian Noise Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
        
        # B) Serial Correlation
        st.markdown("**B) Serial Correlation (Memory)**")
        
        # Calculate correlation between consecutive samples
        if len(brownian_wave.ys) > 1:
            serial_corr = np.corrcoef(brownian_wave.ys[:-1], brownian_wave.ys[1:])[0, 1]
            
            col1, col2 = st.columns(2)
            col1.metric("Brownian Serial Correlation", f"{serial_corr:.6f}")
            
            if show_comparison and len(white_wave.ys) > 1:
                white_corr = np.corrcoef(white_wave.ys[:-1], white_wave.ys[1:])[0, 1]
                col2.metric("White Serial Correlation", f"{white_corr:.6f}")
            
            st.info("✅ Brownian noise typically shows correlation > 0.999 (very strong dependence)")
        
        # C) Power Spectrum (1/f² relationship)
        st.markdown("**C) Power Spectrum (1/f² Dominance)**")
        
        spectrum = brownian_wave.make_spectrum()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot on log-log scale to show 1/f² relationship
        power = np.abs(spectrum.hs) ** 2
        # Avoid log(0)
        valid_idx = (spectrum.fs > 0) & (power > 0)
        
        ax.loglog(spectrum.fs[valid_idx], power[valid_idx], 
                 color='tab:red', linewidth=2, alpha=0.7, label='Brownian Noise')
        
        # Add reference line for 1/f²
        if np.any(valid_idx):
            f_ref = spectrum.fs[valid_idx]
            power_ref = power[valid_idx][0] * (f_ref[0] / f_ref) ** 2
            ax.loglog(f_ref, power_ref, 'k--', alpha=0.3, linewidth=1, label='1/f² Reference')
        
        ax.set_xlabel("Frequency (Hz) [log scale]")
        ax.set_ylabel("Power [log scale]")
        ax.set_title("Power Spectrum: Low-Frequency Dominance")
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        
        st.success("🟤 **Observation**: Power drops rapidly with frequency (1/f² relationship)")
