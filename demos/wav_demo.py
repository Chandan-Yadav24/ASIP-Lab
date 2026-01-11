# demos/wav_demo.py
import streamlit as st
import numpy as np
import scipy.io.wavfile as wav
import io
import matplotlib.pyplot as plt

from .dsp_utils import Wave, read_wave

# --- Main Demo Run Function --------------------------------------------------
def run():
    st.subheader("Reading and Writing WAV Files (Waves) in DSP")

    # 1. Theory Section
    with st.expander("📝 Theory: Reading and Writing Waves", expanded=False):
        st.markdown(
            r"""
            **“Reading and writing Waves”** means loading audio from a WAV file into your program, 
            working with it as digital samples, and then saving it back to a WAV file.

            ### 1) Reading Waves (importing audio)
            **Reading a WAV file** = taking audio stored on disk (like `input.wav`) and converting it 
            into a form your program can work with.
            
            **In `thinkdsp`:**
            ```python
            thinkdsp.read_wave(filename)
            ```
            - Reads a WAV file from disk
            - Returns a `Wave` object

            ### 2) Writing and Playing Waves
            **A) Writing (saving to disk)**
            A `Wave` object provides a `write` method:
            ```python
            wave.write(filename='output.wav')
            ```
            
            **B) Listening / playback**
            - **Play with any media player**: Open generated WAV file.
            - **Programmatic**: `thinkdsp.play_wave()` runs an external player.

            ### 3) What a Wave object actually contains
            A `Wave` represents a digital signal. It typically includes:
            - **`ys`**: NumPy array of **sample values** (amplitudes).
            - **`ts`**: NumPy array of **sample times**.
            - **`framerate`**: **Samples per second** (e.g., 44100).
            
            So you can think of a Wave like:
            > “Here are the amplitudes (`ys`), at these times (`ts`), sampled at this rate (`framerate`).”
            """
        )

    st.markdown("---")

    # 2. Interactive Section
    col_controls, col_viz = st.columns([1, 2])
    
    with col_controls:
        st.write("### 1. Input Source")
        
        # Option to create synthetic signal if no file
        source_mode = st.radio("Choose source:", ["Generate Synthetic Audio", "Upload WAV File"])
        
        wave_obj = None

        if source_mode == "Generate Synthetic Audio":
             freq = st.slider("Frequency (Hz)", 100, 1000, 440)
             duration = st.slider("Duration (s)", 0.5, 5.0, 1.0)
             fs = 16000 # fixed for simple demo
             
             # Generate
             t = np.linspace(0, duration, int(fs * duration), endpoint=False)
             y = np.cos(2 * np.pi * freq * t) * 0.5
             wave_obj = Wave(y, t, fs) # Create our Wave object
             
             st.info("Generated a 'virtual' input.wav in memory.")

        else:
            uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])
            if uploaded_file:
                wave_obj = read_wave(uploaded_file)
    
    with col_viz:
        st.write("### 2. Wave Object Inspection")
        
        if wave_obj:
            # Display properties
            st.success("`thinkdsp.read_wave(...)` successful!")
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Framerate (Hz)", wave_obj.framerate)
            m2.metric("Duration (s)", f"{wave_obj.ts[-1]:.2f}")
            m3.metric("Num Samples", len(wave_obj.ys))

            # Code simulation view
            st.markdown("#### Internal arrays")
            st.code(f"""
# Represents the Wave object contents:
wave.framerate = {wave_obj.framerate}
wave.ts = {str(wave_obj.ts[:3])[:-1]} ... ]  # Start times
wave.ys = {str(wave_obj.ys[:3])[:-1]} ... ]  # Amplitudes
            """, language="python")

            # Plot
            st.pyplot(wave_obj.plot())
            
            st.markdown("### 3. Output (Write & Play)")
            
            # "Write" simulation
            # We treat writing as preparing a bytes buffer for download/play
            buffer = wave_obj.get_audio_bytes()
            
            st.audio(buffer, format='audio/wav')
            
            st.download_button(
                label="Download 'output.wav' (wave.write)",
                data=buffer,
                file_name="output.wav",
                mime="audio/wav"
            )

        else:
            st.info("Select a source to 'read' a Wave.")
