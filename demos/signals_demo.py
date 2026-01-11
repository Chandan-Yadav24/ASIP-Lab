# demos/signals_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def run():
    st.subheader("Signals – Interactive Exploration")

    # --- Theory section in expander -----------------------------------------
    with st.expander("📝 What is a signal? (Theory Overview)", expanded=False):
        st.markdown(
            r"""
A **signal** is anything that carries information. In DSP, a signal is usually a
physical quantity that varies with an **independent variable**, such as:

- **Time** (most common): speech, music, sensor readings  
- **Space**: image brightness over pixels, elevation over a map  
- **Other variables**: temperature, pressure, etc.

So, a signal is basically **information written as a changing value**.

### Why signals matter in the digital world

Signals are everywhere in modern technology:

- Music/audio (microphone → digital samples)  
- Pictures/images (2D arrays of pixel values)  
- Wireless communication (radio waves carrying data)  
- Video (sequence of images vs time)

We constantly record, transmit, store, and modify signals — that’s why DSP is central.

### What DSP does with signals

Digital Signal Processing uses:

- **Mathematics** (to model and analyze signals)  
- **Computers** (to process large amounts of data efficiently)

Two major goals:

1. **Understand / analyze** signals  
   - Find patterns, frequencies, important features  
   - Measure properties (energy, spectrum, correlation, etc.)

2. **Improve / modify** signals  
   - Remove noise  
   - Enhance important parts  
   - Compress for storage/transmission  
   - Filter (smooth or sharpen)  
   - Detect or extract information (e.g., speech recognition, channel decoding)

### Dimensionality of signals

Signals can live in different dimensions:

- **1D signals** (vary over time): audio waveform, ECG, temperature vs time  
- **2D signals** (vary over space): images, maps, elevation surfaces  
- **3D+ signals**: 
  - Video: image vs \(x, y, t\)  
  - Medical volumes: 3D scans  
  - Multi-sensor: space + time + sensor index

### Signals in programming (Signal objects)

In code, a signal is often a **class/object** that represents a mathematical function:

- A `Signal` object can generate values for given inputs (time, space, etc.).  
- Specific signals like **Sinusoid** can be child classes of `Signal`.

This keeps different signal types organized under a common interface.
"""
        )

    # --- Main Layout: Columns -----------------------------------------------
    # Left column for controls, Right column for Visualization
    col_controls, col_viz = st.columns([1, 2])

    with col_controls:
        st.subheader("Signal Configuration")

        dim = st.radio(
            "Signal dimension (conceptual)",
            options=["1D (time signal)", "2D (image-like)"],
            index=0,
        )

        st.markdown("---")

        # Initialize variables to ensure they exist for the viz column
        fs = 500
        duration = 1.0
        signal_type = "Pure sinusoid"
        image_type = "Checkerboard pattern"
        size = 128
        
        # 1D Params defaults
        f1, A1, phi1, phi1_deg = 5.0, 1.0, 0.0, 0
        f2, A2, phi2, phi2_deg = 12.0, 0.7, np.deg2rad(90), 90

        if dim == "1D (time signal)":
            st.markdown("### 1D Time Signal Options")

            signal_type = st.selectbox(
                "Signal type",
                options=[
                    "Pure sinusoid",
                    "Sum of two sinusoids",
                    "Step signal",
                    "Ramp",
                    "Random noise",
                ],
                index=0,
            )

            duration = st.slider(
                "Duration (seconds)", 0.1, 2.0, 1.0, 0.1
            )
            fs = st.slider(
                "Sampling frequency fₛ (Hz)", 50, 2000, 500, 50
            )

            if "sinusoid" in signal_type:
                f1 = st.slider("Frequency f₁ (Hz)", 1.0, 50.0, 5.0, 1.0)
                A1 = st.slider("Amplitude A₁", 0.1, 2.0, 1.0, 0.1)
                phi1_deg = st.slider("Phase φ₁ (degrees)", 0, 360, 0, 15)
                phi1 = np.deg2rad(phi1_deg)

                if signal_type == "Sum of two sinusoids":
                    f2 = st.slider("Frequency f₂ (Hz)", 1.0, 50.0, 12.0, 1.0)
                    A2 = st.slider("Amplitude A₂", 0.1, 2.0, 0.7, 0.1)
                    phi2_deg = st.slider("Phase φ₂ (degrees)", 0, 360, 90, 15)
                    phi2 = np.deg2rad(phi2_deg)

        else:
            st.markdown("### 2D Image-like Signal Options")

            image_type = st.selectbox(
                "2D signal type",
                options=[
                    "Checkerboard pattern",
                    "Horizontal gradient",
                    "Vertical stripes (1D signal over x)",
                ],
                index=0,
            )
            size = st.slider("Image size (N x N)", 32, 256, 128, 32)


    with col_viz:
        # --- Generate and show signals ------------------------------------------
        if dim == "1D (time signal)":
            # Time axis
            N = int(fs * duration)
            n = np.arange(N)
            t = n / fs

            # Generate selected time signal
            if signal_type == "Pure sinusoid":
                x = A1 * np.cos(2 * np.pi * f1 * t + phi1)
                descr = f"Pure sinusoid, f₁ = {f1} Hz, A₁ = {A1}, φ₁ = {phi1_deg}°"
            elif signal_type == "Sum of two sinusoids":
                x1 = A1 * np.cos(2 * np.pi * f1 * t + phi1)
                x2 = A2 * np.cos(2 * np.pi * f2 * t + phi2)
                x = x1 + x2
                descr = (
                    f"Sum of sinusoids: f₁ = {f1} Hz, A₁ = {A1}; "
                    f"f₂ = {f2} Hz, A₂ = {A2}"
                )
            elif signal_type == "Step signal":
                x = np.ones_like(t)
                x[t < duration / 2] = 0.0
                descr = "Step signal (0 then 1)"
            elif signal_type == "Ramp":
                x = (t / duration).astype(float)
                descr = "Ramp signal increasing from 0 to 1"
            elif signal_type == "Random noise":
                x = np.random.randn(N)
                descr = "Random noise (Gaussian samples)"
            else:
                x = np.zeros_like(t)
                descr = "Unknown signal type"

            st.markdown("### 1D Time Signal View")
            st.markdown(f"**Description:** {descr}")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(t, x, color="tab:blue")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.set_title("1D Signal x[n] as a function of time")

            fig.tight_layout()
            st.pyplot(fig)

            st.markdown(
                r"""
    **Interpretation:**

    - This is a **1D signal**: amplitude vs time \(t\).  
    - In your notes, this corresponds to **signals defined over a 1D independent variable** (time).  
    - Different shapes (sinusoidal, step, ramp, noise) represent different **types** of signals.
    """
            )

        else:
            # 2D "image-like" signals
            N = size
            x = np.zeros((N, N), dtype=float)
            descr = ""

            xv, yv = np.meshgrid(np.arange(N), np.arange(N))

            if image_type == "Checkerboard pattern":
                x = ((xv // (N // 8) + yv // (N // 8)) % 2).astype(float)
                descr = "2D checkerboard pattern (alternating blocks)."
            elif image_type == "Horizontal gradient":
                x = np.tile(np.linspace(0, 1, N), (N, 1))
                descr = "Horizontal gradient (brightness changes in x)."
            elif image_type == "Vertical stripes (1D signal over x)":
                freq = 8
                x = 0.5 * (1 + np.sign(np.sin(2 * np.pi * freq * xv / N)))
                descr = "Vertical stripe pattern (1D periodic signal over x)."
            else:
                x = np.zeros((N, N), dtype=float)
                descr = "Unknown image type."

            st.markdown("### 2D Image-like Signal View")
            st.markdown(f"**Description:** {descr}")

            fig2, ax2 = plt.subplots(figsize=(5, 5))
            im = ax2.imshow(x, cmap="gray", origin="lower")
            ax2.set_title("2D Signal (Image-like)")
            ax2.set_xticks([])
            ax2.set_yticks([])
            fig2.colorbar(im, ax=ax2, shrink=0.7)

            fig2.tight_layout()
            st.pyplot(fig2)

            st.markdown(
                r"""
    **Interpretation:**

    - This is a **2D signal**: value vs spatial coordinates \((x, y)\).  
    - In your notes, this corresponds to **images** or other spatial signals.  
    - The independent variable is now **space**, not time.

    - The "Vertical stripes" option shows how a 1D periodic signal in space (over x) becomes a 2D image pattern.
    """
            )

    # --- Summary connection to theory ---------------------------------------
    st.markdown(
        r"""
### Connection back to the theory (Signals section)

- In 1D mode, you are seeing signals like **audio waveforms or sensor readings** vs time.  
- In 2D mode, you are seeing **image-like signals** vs space \((x, y)\).  
- DSP treats all of these as **signals**:
  - Quantities that vary with one or more independent variables.
  - Objects that can be **analyzed**, **filtered**, **compressed**, and **understood** using mathematical tools.
"""
    )