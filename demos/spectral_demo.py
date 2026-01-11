# demos/spectral_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def run():
    st.subheader("Spectral Decomposition – Interactive Demo")

    with st.expander("📝 Mathematical Theory", expanded=False):
        st.markdown(
            r"""
            **Goal:** See how a signal made from a few **sinusoids** looks in **time**
            and how its **frequency spectrum** (DFT) reveals the underlying components.

            We build a signal:

            $$
            x[n] = \sum_{i=1}^{M} A_i \cos\left(2 \pi f_i \frac{n}{f_s} + \varphi_i\right)
            $$

            Where:
            - $A_i$: amplitude of component $i$
            - $f_i$: frequency (Hz) of component $i$
            - $\varphi_i$: phase (radians) of component $i$
            - $f_s$: sampling frequency
            """
        )

    # --- Controls ------------------------------------------------------------
    st.markdown("### Signal settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        fs = st.slider("Sampling frequency fₛ (Hz)", 100, 2000, 500, 50)
    with col2:
        duration = st.slider("Signal duration (seconds)", 0.1, 1.0, 0.5, 0.1)
    with col3:
        num_components = st.selectbox(
            "Number of sinusoidal components (M)",
            options=[1, 2, 3, 4],
            index=1,
        )

    st.markdown("---")
    st.markdown("### Components (frequency, amplitude, phase)")

    N = int(fs * duration)
    n = np.arange(N)
    t = n / fs

    # Prepare arrays
    x_components = []
    component_params = []

    # UI for each component
    # We will use columns to show components side-by-side if they fit, or rows.
    # Given potentially 4 components, let's just stack them in pairs or rows for clarity.
    # Actually, let's simply loop.
    
    for i in range(num_components):
        st.markdown(f"**Component {i+1}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            f_i = st.slider(
                f"f{i+1} (Hz)",
                1.0,
                fs / 2.0,
                float(5 * (i + 1)),
                1.0,
                key=f"freq_{i}",
            )
        with c2:
            A_i = st.slider(
                f"A{i+1} (amplitude)",
                0.1,
                2.0,
                1.0 if i == 0 else 0.7,
                0.1,
                key=f"amp_{i}",
            )
        with c3:
            phi_deg_i = st.slider(
                f"φ{i+1} (degrees)",
                0,
                360,
                0 if i == 0 else 45 * i,
                15,
                key=f"phase_{i}",
            )
        
        phi_i = np.deg2rad(phi_deg_i)

        x_i = A_i * np.cos(2 * np.pi * f_i * t + phi_i)
        x_components.append(x_i)
        component_params.append((f_i, A_i, phi_i))

    # Sum of components
    if x_components:
        x_sum = np.sum(np.stack(x_components, axis=0), axis=0)
    else:
        x_sum = np.zeros_like(t)

    # --- Time-domain plots ---------------------------------------------------
    st.markdown("### Time-domain view")

    fig_time, ax_time = plt.subplots(figsize=(7, 4))

    # Plot each component faintly
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, x_i in enumerate(x_components):
        f_i, A_i, phi_i = component_params[i]
        ax_time.plot(
            t,
            x_i,
            color=colors[i % len(colors)],
            alpha=0.4,
            label=f"Component {i+1}: f={f_i:.1f} Hz, A={A_i:.2f}",
        )

    # Plot sum strongly
    ax_time.plot(t, x_sum, color="black", linewidth=2, label="Sum x[n]")
    ax_time.set_xlabel("Time (seconds)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Signal as sum of sinusoids")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="upper right", fontsize=8)

    fig_time.tight_layout()
    st.pyplot(fig_time)

    # --- Frequency-domain (DFT) ---------------------------------------------
    st.markdown("### Frequency-domain view (Magnitude Spectrum via DFT)")

    # Compute DFT via FFT
    X = np.fft.fft(x_sum)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)

    # Only keep non-negative frequencies (0 to Nyquist)
    idx = freqs >= 0
    freqs_pos = freqs[idx]
    mag_pos = np.abs(X[idx]) / N  # normalize

    fig_freq, ax_freq = plt.subplots(figsize=(7, 4))
    ax_freq.stem(
        freqs_pos,
        mag_pos,
        linefmt="tab:purple",
        markerfmt=" ",
        basefmt=" ",
    )
    ax_freq.set_xlim(0, fs / 2)
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude |X[k]|")
    ax_freq.set_title("Magnitude Spectrum (DFT of x[n])")
    ax_freq.grid(True, alpha=0.3)

    fig_freq.tight_layout()
    st.pyplot(fig_freq)

    # --- Explanation ---------------------------------------------------------
    st.markdown(
        r"""
### How this connects to spectral decomposition

- You built a signal in the **time domain** by adding sinusoids:
  \[
  x[n] = \sum_i A_i \cos\left(2 \pi f_i \frac{n}{f_s} + \varphi_i\right)
  \]

- The **DFT / FFT** of this signal computes its **spectrum**:
  - Peaks in the magnitude spectrum appear near the chosen frequencies \(f_i\).
  - The **height** of each peak relates to the amplitude \(A_i\).
  - The **phase** \(\varphi_i\) affects how the sinusoids line up in time, but does not change the magnitude peaks.

- This is exactly **spectral decomposition**:
  - Representing a discrete-time signal as a sum of sinusoidal components.
  - The spectrum \(X[k]\) tells you which frequencies are present and how strong they are.
"""
    )