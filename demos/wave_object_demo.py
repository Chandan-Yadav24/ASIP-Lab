# demos/wave_object_demo.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .dsp_utils import Wave

def run():
    st.subheader("Wave Objects In-Depth")

    # --- Theory Section (Expander) -------------------------------------------
    with st.expander("📝 Theory: Wave Object Internals", expanded=False):
        st.markdown(
            r"""
            **A Wave Object** is the core data structure for sampled signals in DSP libraries.
            
            ### 1. The 3 Core Attributes
            1. **`ys`**: Array of sample values (amplitudes).
            2. **`ts`**: Array of time points for those samples.
            3. **`framerate`**: Samples per second ($F_s$).
            
            ### 2. Properties
            - **`start`**, **`end`**: First and last time points.
            - **`duration`**: Total length in seconds ($N / F_s$).
            
            ### 3. Modifying a Wave
            - **Scale**: Multiply `ys` by a factor (makes it louder/quieter).
            - **Shift**: Add time offset to `ts` (delays the signal).
            """
        )

    st.markdown("---")

    col_create, col_inspect = st.columns([1, 2])

    with col_create:
        st.write("### 1. Create a Tiny Wave")
        st.info("We'll create a tiny wave with just a few samples to see the raw numbers.")
        
        N = st.slider("Number of samples (N)", 5, 20, 10)
        framerate = st.number_input("Framerate (Hz)", value=10, min_value=1)
        freq = 1.0 
        
        # Create tiny wave
        ts = np.arange(N) / framerate
        ys = np.cos(2 * np.pi * freq * ts)
        wave = Wave(ys, ts, framerate)
        
        st.markdown("### 3. Modify Wave")
        scale_factor = st.slider("Scale Factor (Amplitude)", 0.0, 2.0, 1.0, 0.1)
        time_shift = st.slider("Time Shift (seconds)", -1.0, 1.0, 0.0, 0.1)
        
        # Apply modifications
        # We modify a copy conceptually or just re-gen for purity, 
        # but here we can modify in place as per DSP style
        if scale_factor != 1.0:
            wave.scale(scale_factor)
        if time_shift != 0.0:
            wave.shift(time_shift)

    with col_inspect:
        st.write("### 2. Inspect Internals (ys & ts)")
        
        # Display Properties
        m1, m2, m3 = st.columns(3)
        m1.metric("Start Time", f"{wave.start:.2f} s")
        m2.metric("End Time", f"{wave.end:.2f} s")
        m3.metric("Duration", f"{wave.duration:.2f} s")

        # Display Dataframe of samples
        st.markdown("**Raw Data (Frame vs Sample)**")
        df = pd.DataFrame({
            "Index (n)": range(len(wave.ys)),
            "Time (ts)": wave.ts,
            "Value (ys)": wave.ys
        })
        st.dataframe(df, height=300, use_container_width=True)
        
        # Plot
        st.markdown("**Visual Representation**")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.stem(wave.ts, wave.ys, linefmt='b-', markerfmt='bo', basefmt=' ')
        ax.plot(wave.ts, wave.ys, 'b-', alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform Samples")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown(
            """
            **Observe**:
            - **Scale**: Changes the height of the stems (`ys`).
            - **Shift**: Changes the X-axis values (`ts`), moving the stems left/right.
            """
        )
