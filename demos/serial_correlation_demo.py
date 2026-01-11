# demos/serial_correlation_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Serial Correlation: The Signal's Memory")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Serial Correlation (Lagged Autocorrelation)", expanded=False):
        st.markdown(
            r"""
            **Serial Correlation** measures how a signal relates to itself at a specific delay (**lag**).
            
            ### 1) Mechanics
            To find the correlation at lag $k$:
            1. Take the original samples $x[n]$.
            2. Shift them by $k$ to get $x[n+k]$.
            3. Calculate the Pearson correlation coefficient $\rho$ between the two.
            
            ### 2) Interpreting $\rho$
            - $\rho \approx 1$: Strong "memory". If $x[n]$ is high, $x[n+k]$ is likely high.
            - $\rho \approx 0$: No correlation. Teleporting randomly.
            - $\rho \approx -1$: Strong opposite movement.
            
            ### 3) Noise Model signatures (Lag = 1)
            | Noise Type | Correlation ($\rho$) | Memory |
            |---|---|---|
            | **Uncorrelated (White)** | $\approx 0$ | None |
            | **Pink (1/f)** | $\approx 0.85$ | Moderate |
            | **Brownian (Red)** | $> 0.999$ | High |
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Generate Signal")
        source = st.selectbox(
            "Noise Model", 
            ["White Noise (Uncorrelated)", "Pink Noise (1/f)", "Brownian Noise (Red)"],
            key="ser_corr_source"
        )
        
        lag = st.slider("Lag (k)", 1, 50, 1, key="ser_corr_lag")
        n_samples = st.slider("Samples to View", 500, 5000, 1000, key="ser_corr_samples")
        
        # Generation
        if source == "White Noise (Uncorrelated)":
            sig = dsp_utils.UncorrelatedGaussianNoise(amp=1.0)
            desc = "Teleporting: Current position tells nothing about the next."
        elif source == "Pink Noise (1/f)":
            sig = dsp_utils.PinkNoise(amp=1.0)
            desc = "Walking with purpose: Connected but changing."
        else:
            sig = dsp_utils.BrownianNoise(amp=1.0)
            desc = "Random walk: Next step depends heavily on the current position."
            
        wave = sig.make_wave(duration=n_samples/11025, framerate=11025)
        ys = wave.ys
        
        # Compute Correlation at specific lag
        # shifted_ys = ys[lag:] vs ys[:-lag]
        if lag < len(ys):
            y1 = ys[:-lag]
            y2 = ys[lag:]
            rho = np.corrcoef(y1, y2)[0, 1]
            
            st.metric("Correlation at Lag $k$", f"{rho:.4f}")
            st.write(f"**Analogy**: {desc}")
        
    with col_viz:
        st.write("### 2. Visual Analysis")
        
        # A) Lagged Scatter Plot
        st.markdown(f"**Lagged Scatter Plot** ($x[n]$ vs $x[n+{lag}]$)")
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 5))
        ax_scatter.scatter(y1, y2, alpha=0.3, color='tab:green', s=10)
        ax_scatter.set_xlabel("Sample $x[n]$")
        ax_scatter.set_ylabel(f"Sample $x[n+{lag}]$")
        ax_scatter.set_title(f"Scatter at Lag {lag} (ρ ≈ {rho:.3f})")
        ax_scatter.grid(alpha=0.2)
        st.pyplot(fig_scatter)
        
        # B) Time Series
        st.markdown("**Waveform (Sample View)**")
        fig_time, ax_time = plt.subplots(figsize=(10, 3))
        # Show a slice for clarity
        slice_len = 100
        ax_time.plot(ys[:slice_len], marker='o', color='tab:blue', alpha=0.6)
        # Shifted version
        ax_time.plot(ys[lag:slice_len+lag], marker='x', color='tab:orange', alpha=0.6, linestyle='--')
        ax_time.legend(["Original", f"Shifted (Lag {lag})"])
        ax_time.set_title(f"Comparing Adjacent Steps")
        ax_time.grid(alpha=0.2)
        st.pyplot(fig_time)
        
        st.caption("Notice how 'connected' the blue and orange points look as ρ increases.")
