# demos/correlation_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Correlation: Measuring Similarity")

    # --- Theory Section ------------------------------------------------------
    with st.expander("📝 Theory: Correlation in DSP and Imaging", expanded=False):
        st.markdown(
            r"""
            **Correlation** measures how strongly two signals or variables change together.
            
            ### 1) Pearson Correlation Coefficient ($\rho$)
            Measures linear relationship from -1 to +1.
            - $\rho = 1$: Perfect positive relationship.
            - $\rho = -1$: Perfect negative relationship.
            - $\rho \approx 0$: No linear relationship (independent).
            
            ### 2) Signal Processing Types
            - **Serial Correlation**: Similarity between a sample and its neighbor (detects randomness).
            - **Autocorrelation**: Signal vs. delayed version of itself. Used for **pitch/period detection**.
            - **Cross-correlation**: Signal A vs. shifted Signal B. Used to find **time delays (lags)**.
            
            ### 3) Image Processing
            - **Template Matching**: Sliding a template over an image. High correlation points indicates a match.
            - **Correlation vs Convolution**: Correlation uses the kernel as-is; Convolution flips it.
            """
        )

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Pearson ρ", "Autocorrelation", "Cross-correlation"])

    # --- Tab 1: Pearson Correlation -------------------------------------------
    with tab1:
        st.write("### 1. Pearson Correlation Visualizer")
        st.markdown("Observe how data points align as the correlation coefficient changes.")
        
        rho = st.slider("Target Correlation (ρ)", -1.0, 1.0, 0.7, 0.1, key="corr_rho_slider")
        
        # Generate correlated variables
        # Using Cholesky decomposition idea: Y = rho*X + sqrt(1-rho^2)*Z
        n_points = 500
        x = np.random.normal(0, 1, n_points)
        z = np.random.normal(0, 1, n_points)
        y = rho * x + np.sqrt(1 - rho**2) * z
        
        actual_rho = np.corrcoef(x, y)[0, 1]
        
        fig_rho, ax_rho = plt.subplots(figsize=(6, 5))
        ax_rho.scatter(x, y, alpha=0.5, color='tab:blue', s=20)
        ax_rho.set_xlabel("Variable X")
        ax_rho.set_ylabel("Variable Y")
        ax_rho.set_title(f"Scatter Plot (Actual ρ ≈ {actual_rho:.2f})")
        ax_rho.grid(alpha=0.3)
        ax_rho.set_xlim(-4, 4)
        ax_rho.set_ylim(-4, 4)
        st.pyplot(fig_rho)
        
        st.info("Notice how the cloud of points collapses to a line as ρ approaches 1 or -1.")

    # --- Tab 2: Autocorrelation -----------------------------------------------
    with tab2:
        st.write("### 2. Autocorrelation: Finding the Period")
        st.markdown("Autocorrelation helps find repeating patterns (periodicity) in a signal.")
        
        freq = st.slider("Signal Frequency (Hz)", 50, 500, 200, key="corr_auto_freq")
        noise_level = st.slider("Noise Level", 0.0, 2.0, 0.5, key="corr_auto_noise")
        
        sig = dsp_utils.Sinusoid(freq=freq)
        noise = dsp_utils.WhiteNoise(amp=noise_level)
        signal = sig + noise
        
        wave = signal.make_wave(duration=0.1, framerate=11025)
        
        # Compute autocorrelation manually or using np.correlate
        # R(k) = sum(y[n] * y[n+k])
        # We'll use the "biased" estimator common in DSP
        ys_norm = wave.ys - np.mean(wave.ys)
        lags = np.arange(-len(ys_norm) + 1, len(ys_norm))
        corr = np.correlate(ys_norm, ys_norm, mode='full')
        corr = corr / (np.var(ys_norm) * len(ys_norm)) # Normalization
        
        # Show only positive lags
        half_idx = len(lags) // 2
        lags_pos = lags[half_idx:]
        corr_pos = corr[half_idx:]
        
        col_sig, col_corr = st.columns(2)
        
        with col_sig:
            st.markdown("**Waveform (Noisy Signal)**")
            st.pyplot(wave.plot())
            
        with col_corr:
            st.markdown("**Autocorrelation**")
            fig_ac, ax_ac = plt.subplots(figsize=(10, 6))
            ax_ac.plot(lags_pos, corr_pos, color='tab:purple')
            ax_ac.set_xlabel("Lag (samples)")
            ax_ac.set_ylabel("Correlation")
            ax_ac.set_title("Autocorrelation Function")
            ax_ac.grid(alpha=0.3)
            st.pyplot(fig_ac)
            
        # Detect peak
        # Skip the zero-lag peak
        period_samples = int(wave.framerate / freq)
        st.info(f"The first major peak (after lag 0) should be at approximately **{period_samples}** samples.")

    # --- Tab 3: Cross-correlation ---------------------------------------------
    with tab3:
        st.write("### 3. Cross-correlation: Delay Estimation")
        st.markdown("Find the time shift (lag) between two identical signals.")
        
        true_delay_ms = st.slider("Induced Delay (ms)", 0, 50, 20, key="corr_cross_delay")
        true_delay_samples = int(true_delay_ms * 11025 / 1000)
        
        # Base signal
        base_sig = dsp_utils.Sinusoid(freq=440)
        w1 = base_sig.make_wave(duration=0.1, framerate=11025)
        
        # Delayed signal
        w2_ys = np.roll(w1.ys, true_delay_samples)
        # Add some independent noise to each
        w1.ys = w1.ys + np.random.normal(0, 0.1, len(w1.ys))
        w2_ys = w2_ys + np.random.normal(0, 0.1, len(w2_ys))
        
        # Cross correlate
        xcorr = np.correlate(w1.ys - np.mean(w1.ys), w2_ys - np.mean(w2_ys), mode='full')
        lags = np.arange(-len(w1.ys) + 1, len(w1.ys))
        
        estimated_lag = lags[np.argmax(xcorr)]
        
        fig_cc, (ax_waves, ax_xcorr) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax_waves.plot(w1.ts[:500], w1.ys[:500], label="Signal 1", alpha=0.7)
        ax_waves.plot(w1.ts[:500], w2_ys[:500], label="Signal 2 (Delayed)", alpha=0.7)
        ax_waves.set_title("Comparison of Two Signals")
        ax_waves.legend()
        ax_waves.grid(alpha=0.2)
        
        ax_xcorr.plot(lags, xcorr, color='tab:orange')
        ax_xcorr.set_title(f"Cross-correlation (Peak at Lag: {estimated_lag})")
        ax_xcorr.set_xlabel("Lag (samples)")
        ax_xcorr.axvline(estimated_lag, color='red', linestyle='--')
        ax_xcorr.grid(alpha=0.2)
        
        plt.tight_layout()
        st.pyplot(fig_cc)
        
        st.success(f"**Estimated Time Delay**: {estimated_lag / 11025 * 1000:.2f} ms (Actual: {true_delay_ms} ms)")
