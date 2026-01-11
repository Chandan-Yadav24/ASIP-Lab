# demos/pink_noise_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Pink Noise: The Natural Balance (1/f)")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Pink Noise and the β Parameter", expanded=False):
        st.markdown(
            r"""
            **Pink Noise**: A random signal with power that decreases as frequency increases.
            
            ### 1) Mathematical Definition
            Power at frequency $f$ follows:
            $$P(f) = \frac{K}{f^\beta}$$
            
            Where:
            - $K$: constant (overall level)
            - $\beta$: spectral exponent (controls frequency balance)
            
            ### 2) The β Parameter Spectrum
            - **β = 0**: White noise (flat power, $P(f) = K$)
            - **β = 1**: Pink noise (1/f, balanced across octaves)
            - **β = 2**: Brownian/Red noise (1/f², very low-frequency heavy)
            - **0 < β < 2**: Intermediate "colored" noise
            
            ### 3) Characteristics
            **Serial Correlation** (memory between samples):
            - White (β=0): ~0 (no memory)
            - Pink (β=1): ~0.851 (moderate memory)
            - Brownian (β=2): ~1 (strong memory)
            
            **Waveform Appearance**:
            - White: Jagged, rapid changes
            - Pink: Some wandering, still random
            - Brownian: Smooth random walk
            
            ### 4) Why "Pink"?
            Called "pink" by analogy to light: energy is more balanced across the spectrum than white noise, 
            similar to how pink light has more red (low frequency) than white light.
            
            ### 5) Sound Quality
            - **White**: Harsh hiss (like static)
            - **Pink**: Natural, balanced (like a waterfall or rain)
            - **Brownian**: Deep rumble (like distant thunder)
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Configure Noise")
        
        beta = st.slider(
            "β (Spectral Exponent)", 
            0.0, 2.0, 1.0, 0.1,
            key="pink_beta",
            help="0=White, 1=Pink, 2=Brownian"
        )
        
        # Display noise type based on beta
        if beta < 0.3:
            noise_label = "White-ish Noise"
            color = "🤍"
        elif beta < 0.7:
            noise_label = "Light Pink Noise"
            color = "🩷"
        elif beta < 1.3:
            noise_label = "Pink Noise"
            color = "🎨"
        elif beta < 1.7:
            noise_label = "Dark Pink Noise"
            color = "🟣"
        else:
            noise_label = "Brownian-ish Noise"
            color = "🟤"
        
        st.info(f"{color} **{noise_label}** (β = {beta:.1f})")
        
        duration = st.slider("Duration (s)", 0.5, 3.0, 1.5, key="pink_duration")
        
        show_comparison = st.checkbox("Show All Three Types", value=False, key="pink_compare")
        
        framerate = 11025
        
        # Generate noise
        signal = dsp_utils.PinkNoise(amp=0.5, beta=beta)
        wave = signal.make_wave(duration=duration, framerate=framerate)
        
        # Generate comparison signals if needed
        if show_comparison:
            white_signal = dsp_utils.UncorrelatedGaussianNoise(amp=0.5)
            white_wave = white_signal.make_wave(duration=duration, framerate=framerate)
            
            brownian_signal = dsp_utils.BrownianNoise(amp=0.5)
            brownian_wave = brownian_signal.make_wave(duration=duration, framerate=framerate)
        
        st.write("### 3. Listen")
        st.audio(wave.get_audio_bytes(), format='audio/wav')
        st.caption(f"β = {beta:.1f}: Listen to the spectral balance")

    with col_viz:
        st.write("### 2. Analysis")
        
        # A) Waveform
        st.markdown("**A) Waveform: Smoothness vs β**")
        
        if show_comparison:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7))
            segment = min(2000, len(wave.ys))
            
            # White
            ax1.plot(white_wave.ts[:segment], white_wave.ys[:segment], 
                    color='gray', linewidth=1, alpha=0.8)
            ax1.set_title("White Noise (β=0): Jagged")
            ax1.set_ylabel("Amplitude")
            ax1.grid(alpha=0.3)
            
            # Pink (current β)
            ax2.plot(wave.ts[:segment], wave.ys[:segment], 
                    color='hotpink', linewidth=1, alpha=0.8)
            ax2.set_title(f"Pink Noise (β={beta:.1f}): Intermediate")
            ax2.set_ylabel("Amplitude")
            ax2.grid(alpha=0.3)
            
            # Brownian
            ax3.plot(brownian_wave.ts[:segment], brownian_wave.ys[:segment], 
                    color='brown', linewidth=1, alpha=0.8)
            ax3.set_title("Brownian Noise (β=2): Smooth Wandering")
            ax3.set_xlabel("Time (s)")
            ax3.set_ylabel("Amplitude")
            ax3.grid(alpha=0.3)
            
            fig.tight_layout()
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(10, 3))
            segment = min(2000, len(wave.ys))
            ax.plot(wave.ts[:segment], wave.ys[:segment], 
                   color='hotpink', linewidth=1.5, alpha=0.8)
            ax.set_title(f"Pink Noise Waveform (β={beta:.1f})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
        
        # B) Serial Correlation
        st.markdown("**B) Serial Correlation (Memory)**")
        
        if len(wave.ys) > 1:
            serial_corr = np.corrcoef(wave.ys[:-1], wave.ys[1:])[0, 1]
            
            if show_comparison:
                col1, col2, col3 = st.columns(3)
                
                white_corr = np.corrcoef(white_wave.ys[:-1], white_wave.ys[1:])[0, 1]
                brownian_corr = np.corrcoef(brownian_wave.ys[:-1], brownian_wave.ys[1:])[0, 1]
                
                col1.metric("White (β=0)", f"{white_corr:.4f}")
                col2.metric(f"Pink (β={beta:.1f})", f"{serial_corr:.4f}")
                col3.metric("Brownian (β=2)", f"{brownian_corr:.4f}")
            else:
                st.metric(f"Serial Correlation (β={beta:.1f})", f"{serial_corr:.4f}")
            
            # Expected correlation info
            if 0.9 <= beta <= 1.1:
                st.success("✅ For β≈1 (true pink), expect correlation around 0.85")
        
        # C) Power Spectrum with 1/f^β overlay
        st.markdown("**C) Power Spectrum: 1/f^β Relationship**")
        
        spectrum = wave.make_spectrum()
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot on log-log scale
        power = np.abs(spectrum.hs) ** 2
        valid_idx = (spectrum.fs > 0) & (power > 0)
        
        ax.loglog(spectrum.fs[valid_idx], power[valid_idx], 
                 color='hotpink', linewidth=2, alpha=0.7, label=f'Pink Noise (β={beta:.1f})')
        
        # Add reference line for 1/f^β
        if np.any(valid_idx):
            f_ref = spectrum.fs[valid_idx]
            power_ref = power[valid_idx][0] * (f_ref[0] / f_ref) ** beta
            ax.loglog(f_ref, power_ref, 'k--', alpha=0.4, linewidth=1.5, 
                     label=f'1/f^{beta:.1f} Reference')
        
        ax.set_xlabel("Frequency (Hz) [log scale]")
        ax.set_ylabel("Power [log scale]")
        ax.set_title(f"Power Spectrum: 1/f^{beta:.1f}")
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("---")
        st.markdown("### 🔍 Interpretation")
        
        if beta < 0.3:
            st.info("📊 **White-ish**: Power is nearly flat across frequencies")
        elif beta < 1.3:
            st.success("🎨 **Pink**: Power decreases steadily with frequency (1/f relationship)")
        else:
            st.warning("🟤 **Brownian-ish**: Power drops steeply, concentrated at low frequencies")
