# demos/periodic_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def run():
    """Interactive periodic signal demo."""
    st.subheader("Experience a Periodic Signal")

    with st.expander("📝 Mathematical Theory", expanded=False):
        st.markdown(r"A continuous-time periodic signal can be defined as:")
        st.latex(r"x(t) = A \cos(2 \pi f t + \varphi)")
        st.markdown(
            r"""
            Where:
            - **$A$** is the amplitude (peak value).
            - **$f$** is the frequency (cycles per second).
            - **$\varphi$** is the phase shift (in radians).
            """
        )

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        A = st.slider("Amplitude A", 0.1, 2.0, 1.0, 0.1)

    with col2:
        f = st.slider("Frequency f (Hz)", 0.5, 10.0, 2.0, 0.5)

    with col3:
        phi_deg = st.slider("Phase φ (degrees)", 0, 360, 0, 15)
    phi = np.deg2rad(phi_deg)

    # Time axis: show 3 periods
    T0 = 1.0 / f   # period in seconds
    t_end = 3 * T0
    t = np.linspace(0, t_end, 1000)

    x_t = A * np.cos(2 * np.pi * f * t + phi)

    st.markdown(
        r"""
We also **sample** this continuous-time signal to get a discrete-time
signal \(x[n] = x(nT_s)\).

Use the slider below to choose a sampling frequency \(f_s\).
"""
    )

    fs = st.slider("Sampling frequency fₛ (Hz)", 4.0, 40.0, 16.0, 2.0)
    Ts = 1.0 / fs

    n = np.arange(0, int(t_end * fs))
    t_samples = n * Ts
    x_n = A * np.cos(2 * np.pi * f * t_samples + phi)

    # Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    # Continuous-time signal
    ax1.plot(t, x_t, color="tab:blue")
    ax1.set_title("Continuous-time periodic signal x(t)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)

    # Discrete-time samples
    ax2.plot(t, x_t, color="lightgray", linewidth=1)
    ax2.stem(
        t_samples,
        x_n,
        linefmt="tab:orange",
        markerfmt="o",
        basefmt=" "
    )
    ax2.set_title("Sampled signal x[n] = x(nTₛ)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)

    st.markdown(
        r"""
### What to observe

- Change **frequency** `f`:
  - Higher `f` → more cycles in the same time (shorter period).
  - Lower `f` → fewer cycles (longer period).

- Change **sampling frequency** `fₛ`:
  - When `fₛ` is **much larger** than `f` (e.g. f = 2 Hz, fₛ = 32 Hz), the samples nicely capture the waveform.
  - If `fₛ` is too low, the samples may not show the shape well → connects to **aliasing** and **Nyquist**.

- Change **phase φ**:
  - The wave shifts left/right, but its shape and period stay the same.

This demo lets you **see and feel** what a periodic signal is in both continuous time
and its sampled (discrete-time) version.
"""
    )