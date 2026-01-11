# demos/signal_object_demo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from . import dsp_utils

def run():
    st.subheader("Signal Objects Hierarchy")

    # --- Theory Section (Expander) -------------------------------------------
    with st.expander("📝 Theory: Signal Objects in DSP", expanded=False):
        st.markdown(
            r"""
            **Signal Object**: A Python representation of a mathematical function (a distinct "rule" for generating a signal).
            
            ### 1) What a Signal object is
            - **Signal** = abstract function (formula).
            - It helps you:
                - Choose time values `ts`.
                - Compute values `ys` at those times.
                - Create a **Wave** (sampled data).
            
            ### 2) Class Hierarchy
            - **`Signal` (Parent)**: Base class. Provides `make_wave()`.
            - **`Sinusoid`**: Represents sine/cosine signals. Defined by `freq`, `amp`, `offset`.
            - **`SumSignal`**: Result of adding signals (e.g., `sig1 + sig2`).
            - **`Noise`**: Generates random values (Uniform, Gaussian, etc.).
            
            ### 3) Key Workflow
            1. **`__init__`**: Create the signal (the "Score").
            2. **`evaluate(times)`**: Calculate values at specific times.
            3. **`make_wave(...)`**: Samples the signal into a **Wave** object (the "Performance").
            
            > **Signal (Math) $\to$ Wave (Samples) $\to$ Spectrum (Frequency)**
            """
        )

    st.markdown("---")

    # --- Interactive Demo ----------------------------------------------------
    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.write("### 1. Define Signal (The Score)")
        
        sig_choice = st.radio("Signal Architecture", ["Pure Sinusoid", "Sum (Sinusoid + Noise)", "Complex Sum (2 Tones)"])
        
        signal = None
        code_str = ""
        
        if sig_choice == "Pure Sinusoid":
            freq = st.slider("Frequency (Hz)", 220, 880, 440)
            amp = st.slider("Amplitude", 0.1, 2.0, 1.0)
            signal = dsp_utils.Sinusoid(freq=freq, amp=amp)
            code_str = f"signal = Sinusoid(freq={freq}, amp={amp})"
            
        elif sig_choice == "Sum (Sinusoid + Noise)":
            freq = st.slider("Sine Frequency (Hz)", 220, 880, 440)
            noise_amp = st.slider("Noise Amplitude", 0.0, 1.0, 0.2)
            
            # Create two signals
            s1 = dsp_utils.Sinusoid(freq=freq, amp=1.0)
            n1 = dsp_utils.WhiteNoise(amp=noise_amp)
            
            # Add them using __add__ logic
            signal = s1 + n1
            code_str = f"s1 = Sinusoid(freq={freq})\nn1 = WhiteNoise(amp={noise_amp})\nsignal = s1 + n1  # SumSignal"
            
        elif sig_choice == "Complex Sum (2 Tones)":
            f1 = st.slider("Freq 1 (Hz)", 200, 500, 300)
            f2 = st.slider("Freq 2 (Hz)", 500, 1000, 600)
            
            s1 = dsp_utils.Sinusoid(freq=f1, amp=1.0)
            s2 = dsp_utils.Sinusoid(freq=f2, amp=0.5)
            
            signal = s1 + s2
            code_str = f"s1 = Sinusoid(freq={f1})\ns2 = Sinusoid(freq={f2})\nsignal = s1 + s2"

        st.code(code_str, language="python")
        
        st.write("### 2. Make Wave (The Performance)")
        duration = st.slider("Duration (s)", 0.5, 3.0, 1.0)
        framerate = 11025 # Fixed for demo perf
        
        if st.button("signal.make_wave()", type="primary"):
            # Create the Wave
            wave = signal.make_wave(duration=duration, framerate=framerate)
            
            # Save to session state to persist across reruns if needed, 
            # though in this simple structure we can just compute it directly.
            st.session_state['last_wave'] = wave
            st.session_state['last_code'] = code_str

    with col_viz:
        st.write("### 3. Analyze Result")
        
        # Check if we have a wave to show
        # (Auto-run logic: if simple app, just run it. The button above is nice for "Action" feel 
        # but in Streamlit we usually just run to show state).
        # Let's just always run it for immediate feedback, the button can just be a "Refresher"
        try:
            wave = signal.make_wave(duration=duration, framerate=framerate)
            
            # Show Wave
            st.markdown(f"**Wave Object** (created via `make_wave`)")
            st.pyplot(wave.plot())
            
            # Play
            st.audio(wave.get_audio_bytes(), format='audio/wav')
            
            # Show Spectrum
            st.markdown("**Spectrum** (derived from Wave)")
            spectrum = wave.make_spectrum()
            st.pyplot(spectrum.plot())
            
            st.info("Notice how the Spectrum confirms the components you defined in the Signal object.")
            
        except Exception as e:
            st.error(f"Error generating wave: {e}")
